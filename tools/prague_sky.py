"""NumPy port of the Prague sky model's sky-radiance evaluation (Wilkie et al. 2021, "A fitted
radiance and attenuation model for realistic atmospheres", ACM TOG 40(4):135; Vevoda et al.
2022, "A Wide Spectral Range Sky Radiance Model", Pacific Graphics).

Ported from ``PragueSkyModel.cpp`` of https://github.com/PetrVevoda/pragueskymodel
(Copyright 2022 Charles University, Apache License 2.0; see ``third_party/licenses/pragueskymodel-LICENSE.txt``).
Only the ground-level sky radiance (``readRadiance``, ``computeParameters``, ``evaluateModel``,
``skyRadiance``) is ported; the transmittance, polarisation and solar-disc parts are not.

The model coefficients are NOT part of open-cam: the dataset (e.g. the 280-2480 nm
``PragueSkyModelDatasetSWIR.dat``) is distributed by the authors without a stated licence, so each
user downloads it from the project page and passes its path (``--prague-dataset`` or the
``OPEN_CAM_PRAGUE_DATASET`` environment variable).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PLANET_RADIUS = 6378000.0
SAFETY_ALTITUDE = 50.0
DATASET_ENV = "OPEN_CAM_PRAGUE_DATASET"


def dataset_path(path: str | os.PathLike | None = None) -> Path:
    """The dataset file from ``path`` or ``$OPEN_CAM_PRAGUE_DATASET``; raises if missing."""
    p = path or os.environ.get(DATASET_ENV)
    if not p:
        raise FileNotFoundError(
            f"Prague sky model dataset not given: pass its path or set {DATASET_ENV} "
            "(download it from https://github.com/PetrVevoda/pragueskymodel)"
        )
    p = Path(p).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"Prague sky model dataset not found: {p}")
    return p


def _interp_param(q, breaks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``getInterpolationParameter``: segment index and factor of ``q`` in ``breaks`` (clamped)."""
    q = np.clip(np.asarray(q, float), breaks[0], breaks[-1])
    nxt = np.searchsorted(breaks[1:], q, side="right") + 1
    idx = nxt - 1
    at_end = nxt >= len(breaks)
    nxt_c = np.minimum(nxt, len(breaks) - 1)
    span = breaks[nxt_c] - breaks[idx]
    factor = np.where(at_end, 0.0, (q - breaks[idx]) / np.where(span == 0, 1.0, span))
    return idx, factor


def _eval_pl(coefs: np.ndarray, idx: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """``evalPL`` on the last axis of ``coefs`` at per-pixel (idx, factor)."""
    hi = np.minimum(idx + 1, coefs.shape[-1] - 1)
    c0 = np.take(coefs, idx, axis=-1)
    c1 = np.take(coefs, hi, axis=-1)
    return (c1 - c0) * factor + c0


@dataclass
class SkyAngles:
    gamma: np.ndarray  # angle view - sun [rad]
    alpha_src: np.ndarray  # zero (or shadow when the sun is below the horizon) [rad]
    zero: np.ndarray  # corrected view zenith angle [rad]


class PragueSky:
    """Ground-level sky radiance from a Prague sky model dataset file (memory-mapped).

    ``radiance`` returns spectral radiance in the dataset's units (W m^-2 sr^-1 nm^-1), one value
    per 40 nm (SWIR dataset) channel, constant within each channel as in the reference code.
    """

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = dataset_path(path)
        with open(self.path, "rb") as f:

            def ri() -> int:
                return struct.unpack("<i", f.read(4))[0]

            def rd(n: int) -> np.ndarray:
                return np.frombuffer(f.read(8 * n), "<f8").copy()

            self.visibilities = rd(ri())
            self.albedos = rd(ri())
            self.altitudes = rd(ri())
            self.elevations = rd(ri())
            self.channels = ri()
            self.channel_start = float(rd(1)[0])
            self.channel_width = float(rd(1)[0])
            self.rank = ri()
            self.sun_breaks = rd(ri())
            self.zenith_breaks = rd(ri())
            self.emph_breaks = rd(ri())
            offset = f.tell()
        ns, nz, ne = len(self.sun_breaks), len(self.zenith_breaks), len(self.emph_breaks)
        rank_dt = np.dtype([("sun", "<f2", (ns,)), ("scale", "<f8"), ("zen", "<f2", (nz,))])
        cfg_dt = np.dtype([("r", rank_dt, (self.rank,)), ("emph", "<f2", (ne,))])
        shape = (
            len(self.visibilities),
            len(self.albedos),
            len(self.altitudes),
            len(self.elevations),
            self.channels,
        )
        self._cfg = np.memmap(self.path, dtype=cfg_dt, mode="r", offset=offset, shape=shape)

    @property
    def channel_edges_nm(self) -> np.ndarray:
        return self.channel_start + self.channel_width * np.arange(self.channels + 1)

    @property
    def channel_centres_nm(self) -> np.ndarray:
        e = self.channel_edges_nm
        return 0.5 * (e[:-1] + e[1:])

    def _config(self, vis: int, alb: int, alt: int, ele: int, ch: int):
        c = self._cfg[vis, alb, alt, ele, ch]
        sun = c["r"]["sun"].astype(np.float64)  # (rank, ns)
        zen = c["r"]["zen"].astype(np.float64) / c["r"]["scale"][:, None]
        return sun, zen, c["emph"].astype(np.float64)

    def _reconstruct(self, cfg, a: tuple) -> np.ndarray:
        sun, zen, emph = cfg
        (gi, gf), (ai, af), (zi, zf) = a
        out = np.zeros(np.shape(gi))
        for r in range(self.rank):
            out += _eval_pl(sun[r], gi, gf) * _eval_pl(zen[r], ai, af)
        if len(self.emph_breaks):
            out = np.maximum(out * _eval_pl(emph, zi, zf), 0.0)
        return out

    def angles(self, view_dirs: np.ndarray, sun_elevation_rad: float, sun_azimuth_rad: float = 0.0) -> tuple:
        """``computeParameters`` for a viewpoint at ground level (z up, sun azimuth from +x)."""
        v = np.asarray(view_dirs, float)
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        d = PLANET_RADIUS + SAFETY_ALTITUDE
        corr = np.sqrt(d * d - PLANET_RADIUS * PLANET_RADIUS) / d
        cv = v.copy()
        cv[..., 2] += corr
        cv /= np.linalg.norm(cv, axis=-1, keepdims=True)
        se, sa = sun_elevation_rad, sun_azimuth_rad
        sun = np.array([np.cos(sa) * np.cos(se), np.sin(sa) * np.cos(se), np.sin(se)])
        sh = se + 0.5 * np.pi
        shadow_dir = np.array([np.cos(sh) * np.cos(sa), np.cos(sh) * np.sin(sa), np.sin(sh)])
        gamma = np.arccos(np.clip(v @ sun, -1.0, 1.0))
        shadow = np.arccos(np.clip(cv @ shadow_dir, -1.0, 1.0))
        zero = np.arccos(np.clip(cv[..., 2], -1.0, 1.0))
        alpha_src = shadow if se < 0.0 else zero
        return (
            _interp_param(gamma, self.sun_breaks),
            _interp_param(alpha_src, self.zenith_breaks),
            _interp_param(zero, self.emph_breaks),
        )

    def radiance(
        self,
        view_dirs: np.ndarray,
        sun_elevation_deg: float,
        visibility_km: float = 50.0,
        albedo: float = 0.1,
        sun_azimuth_deg: float = 0.0,
        channels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sky radiance (..., n_channels) for unit view directions (..., 3) seen from the ground."""
        if not 0.0 <= albedo <= 1.0:
            raise ValueError("albedo must be in [0, 1]")
        a = self.angles(view_dirs, np.radians(sun_elevation_deg), np.radians(sun_azimuth_deg))
        chans = np.arange(self.channels) if channels is None else np.asarray(channels, int)
        globals_ = [
            _interp_param(visibility_km, self.visibilities),
            _interp_param(albedo, self.albedos),
            _interp_param(SAFETY_ALTITUDE, self.altitudes),
            _interp_param(sun_elevation_deg, self.elevations),
        ]
        sizes = [len(self.visibilities), len(self.albedos), len(self.altitudes), len(self.elevations)]
        out = np.empty(np.shape(view_dirs)[:-1] + (len(chans),))
        for k, ch in enumerate(chans):
            out[..., k] = self._interpolate(a, globals_, sizes, int(ch), 0, [])
        return out

    def _interpolate(self, a, globals_, sizes, ch: int, level: int, idx: list) -> np.ndarray:
        """Recursive multilinear interpolation over (visibility, albedo, altitude, elevation)."""
        if level == 4:
            return self._reconstruct(self._config(*idx, ch), a)
        i, f = int(globals_[level][0]), float(globals_[level][1])
        low = self._interpolate(a, globals_, sizes, ch, level + 1, [*idx, i])
        if f < 1e-6:
            return low
        high = self._interpolate(a, globals_, sizes, ch, level + 1, [*idx, min(i + 1, sizes[level] - 1)])
        return (1.0 - f) * low + f * high
