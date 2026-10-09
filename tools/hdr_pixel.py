"""HDR pixel architectures for the EMVA sensor model (opt-in via ``noise.hdr``).

Architectures (``noise.hdr.architecture``):

``dcg``            Dual conversion gain: one photodiode read twice through the floating
                   diffusion at a high (HCG) and a low (LCG) conversion gain.  Per-gain
                   K, read noise and swing (full well); "dual" readout writes both raws,
                   "switch" models an auto-gain-switch (merge = select).
``split_pixel``    Large + small photodiode with a sensitivity ratio (optionally per CFA
                   channel), separate full wells, read noise, PRNU, DSNU and dark current;
                   each photodiode may have one or more readouts (e.g. LPD HCG + LCG).
``lofic``          Lateral overflow integration capacitor: charge above the photodiode
                   full well spills into a capacitor; HCG reads the photodiode (with CDS),
                   LCG reads photodiode + capacitor with its own K, read noise, optional
                   kTC noise of the combined capacitance and capacitor dark current.
``multi_exposure`` N sequential exposures with ratios t_i / t_0 on the same photodiode
                   (independent shot noise, shared PRNU), optionally fed with separate
                   per-exposure electron images (e.g. rendered per window with
                   ``tools/render_time_slices.py --shutter-open-s/--integration-time-s``)
                   so motion and LED-flicker artefacts appear between captures.

Signal model (per collector ``j`` with response ``r_j`` relative to the reference
photodiode/exposure, per readout ``k``)::

    Q_j   = Poisson(r_j mu g_j + D_j + o_j)              g_j = 1 + PRNU_j z, o_j = DSNU_j z'
    Q_pd  = min(Q_j, FW_pd),  Q_ov = clip(Q_j - FW_pd, 0, C_ov)          (LOFIC only)
    q_k   = min(Q_pd [+ Q_ov + D_cap], FW_k) + N(0, sigma_k^2)
    DN_k  = clip(round(q_k / K_k + black_k), 0, 2^b_k - 1)               (EMVA 1288 linear model)

Merge (reference electrons, dark-subtracted).  Each readout gives an unbiased estimate
``mu_k = ((DN_k - black_k) K_k - D_k) / r_j``; readouts whose measured charge exceeds
``threshold_fraction`` x their saturation are discarded (the transition threshold).  The
remaining estimates are combined with the minimum-variance (generalised least squares)
weights ``w = C^-1 1 / (1^T C^-1 1)``, where ``C`` is the temporal covariance of the
estimates evaluated at a pilot signal (shot + dark shot shared between readouts of the same
collector, read noise and K^2/12 quantisation per readout).  This is the noise-optimal
weighting of Granados et al. (CVPR 2010, doi:10.1109/CVPR.2010.5540208) and Hasinoff et
al. (CVPR 2010, doi:10.1109/CVPR.2010.5540167) generalised to correlated readouts of one
photodiode (DCG/LOFIC); ``method: select`` instead takes the lowest-variance valid readout
(an auto-switching sensor).  The analytic SNR (:func:`theory_snr`) uses exactly the same
covariance and thresholds, so it predicts the SNR dips at each transition, i.e. the drop
when the most sensitive capture saturates and the next one takes over (EMVA 1288 Release
4.0 General, Sec. on non-linear/HDR cameras, requires dense exposure steps there).

Optional piece-wise-linear companding (``noise.hdr.compand``) maps linear HDR DN to a
``bit_depth`` code; decompanding is the inverse PWL and contributes ``Delta_in^2/12``
quantisation variance per segment (``Delta_in`` = input DN per output code).

References (all verified):
  EMVA Standard 1288 Release 4.0 Linear and General (2021),
    https://www.emva.org/standards-technology/emva-1288/  (SNR, sigma_q^2 = 1/12 DN^2,
    total SNR with DSNU/PRNU Sec. 8.5; HDR/non-linear cameras in Release 4.0 General).
  A. Darmont, High Dynamic Range Imaging: Sensors and Architectures, 2nd ed., SPIE,
    2019, doi:10.1117/3.2512264 (DCG, split pixel, LOFIC, multi-exposure, SNR dips).
  N. Akahane et al., "A sensitivity and linearity improvement of a 100-dB dynamic range
    CMOS image sensor using a lateral overflow integration capacitor", IEEE JSSC 41(4),
    2006, doi:10.1109/JSSC.2006.870753 (LOFIC).
  N. Akahane, R. Adachi, S. Mizobuchi, S. Sugawa, "Optimum design of conversion gain and
    full well capacity in CMOS image sensor with lateral overflow integration capacitor",
    IEEE TED 56(11), 2009, doi:10.1109/TED.2009.2030550.
  I. Takayanagi et al., "A 120-ke- full-well capacity 160-uV/e- conversion gain 2.8-um
    backside-illuminated pixel with a lateral overflow integration capacitor", Sensors
    19(24):5572, 2019, doi:10.3390/s19245572.
  B. Deegan, "The effect of split pixel HDR image sensor technology on MTF measurements",
    Proc. SPIE 9023, 2014, doi:10.1117/12.2039327 (split pixel).
  I. Takayanagi and R. Kuroda, "HDR CMOS image sensors for automotive applications",
    IEEE TED 69(6), 2022, doi:10.1109/TED.2022.3164370 (DCG, split pixel, LOFIC review).
  M. Innocent et al., "Automotive 8.3 MP CMOS image sensor with 150 dB dynamic range and
    light flicker mitigation", IEDM 2021, doi:10.1109/IEDM19574.2021.9720683.
  Y. Luo and S. Mirabbasi, "A 60fps 9.9nJ/frame-pixel CMOS image sensor with on-chip
    pixel-wise conversion gain modulation for per-frame adaptive DCG-HDR imaging", VLSI
    2023, doi:10.23919/VLSITechnologyandCir57934.2023.10185405 (DCG).
  M. Mase et al., "A wide dynamic range CMOS image sensor with multiple exposure-time
    signal outputs and 12-bit column-parallel cyclic A/D converters", IEEE JSSC 40(12),
    2005, doi:10.1109/JSSC.2005.858477 (multi-exposure).
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

K_B_J_PER_K = 1.380649e-23
Q_E_C = 1.602176634e-19
ARCHITECTURES = ("dcg", "split_pixel", "lofic", "multi_exposure")
_BAYER_PHASE = {"RGGB": (0, 1, 1, 2), "BGGR": (2, 1, 1, 0), "GRBG": (1, 0, 2, 1), "GBRG": (1, 2, 0, 1)}
_CHUNK = 1 << 17


@dataclass(frozen=True)
class Collector:
    """A charge collector: one photodiode during one integration window."""

    name: str
    response: float = 1.0
    full_well_e: float = math.inf
    overflow_capacity_e: float = 0.0
    overflow_dark_e: float = 0.0
    dark_e: float = 0.0
    prnu_std: float = 0.0
    dsnu_std_e: float = 0.0
    photodiode: str = ""
    t_start_s: float = 0.0
    t_int_s: float = 0.0
    response_rgb: tuple[float, float, float] | None = None

    @property
    def pd(self) -> str:
        return self.photodiode or self.name


@dataclass(frozen=True)
class Readout:
    """One digitised capture of a collector (one per-capture raw)."""

    name: str
    collector: str
    K_e_per_DN: float
    sigma_e: float
    full_well_e: float = math.inf
    bit_depth: int = 12
    black_DN: float = 0.0
    reads_overflow: bool = False

    @property
    def max_dn(self) -> float:
        return float((1 << self.bit_depth) - 1)


@dataclass(frozen=True)
class Compander:
    knees_in: tuple[float, ...]
    knees_out: tuple[float, ...]
    bit_depth: int = 12
    black_DN: float = 0.0


@dataclass(frozen=True)
class HdrArchitecture:
    name: str
    collectors: tuple[Collector, ...]
    readouts: tuple[Readout, ...]
    method: str = "snr_weighted"
    threshold_fraction: float = 0.9
    hdr_K_e_per_DN: float | None = None
    compander: Compander | None = None
    notes: dict = field(default_factory=dict)

    def collector(self, name: str) -> Collector:
        for c in self.collectors:
            if c.name == name:
                return c
        raise KeyError(name)

    def readout_dark_e(self, r: Readout) -> float:
        c = self.collector(r.collector)
        return c.dark_e + (c.overflow_dark_e if r.reads_overflow else 0.0)

    def saturation_e(self, r: Readout) -> float:
        """Largest read charge (e-) the readout can report without clipping."""
        c = self.collector(r.collector)
        cap = c.full_well_e + (c.overflow_capacity_e if r.reads_overflow else 0.0)
        return float(min(r.full_well_e, cap, (r.max_dn - r.black_DN) * r.K_e_per_DN))

    @property
    def K_hdr(self) -> float:
        if self.hdr_K_e_per_DN:
            return float(self.hdr_K_e_per_DN)
        return float(min(r.K_e_per_DN / self.collector(r.collector).response for r in self.readouts))

    def transitions_e(self) -> list[tuple[str, float]]:
        """Reference-electron signal at which each readout drops out of the merge."""
        out = []
        for r in self.readouts:
            c = self.collector(r.collector)
            out.append((r.name, (self.threshold_fraction * self.saturation_e(r) - self.readout_dark_e(r)) / c.response))
        return sorted(out, key=lambda t: t[1])

    @property
    def max_reference_e(self) -> float:
        return max(t for _, t in self.transitions_e())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def hdr_enabled(cfg: dict | None) -> bool:
    return bool(((cfg or {}).get("hdr") or {}).get("enabled", False))


def _ktc_sigma_e(capacitance_fF: float, temperature_c: float) -> float:
    if capacitance_fF <= 0:
        return 0.0
    return math.sqrt(K_B_J_PER_K * (temperature_c + 273.15) * capacitance_fF * 1e-15) / Q_E_C


def _readout(name: str, collector: str, d: dict | None, base: dict, *, full_well: float, overflow: bool = False):
    d = d or {}
    sigma = float(d.get("sigma_e", base["sigma_d_e"]))
    ktc = _ktc_sigma_e(float(d.get("ktc_capacitance_fF", 0.0)), float(base.get("temperature_c", 20.0)))
    return Readout(
        name=name,
        collector=collector,
        K_e_per_DN=float(d.get("K_e_per_DN", base["K_e_per_DN"])),
        sigma_e=math.hypot(sigma, ktc),
        full_well_e=float(d.get("full_well_e", full_well)),
        bit_depth=int(d.get("bit_depth", base["bit_depth"])),
        black_DN=float(d.get("black_level_DN", base["black_DN"])),
        reads_overflow=overflow,
    )


def _pd_params(d: dict, base: dict, t_int: float) -> dict:
    rate = float(d.get("dark_current_e_per_s", base["dark_current_e_per_s"]))
    return {
        "dark_e": max(0.0, rate * t_int * float(base.get("dark_temp_scale", 1.0))),
        "prnu_std": float(d.get("prnu_std_fraction", base["prnu_std"])),
        "dsnu_std_e": float(d.get("dsnu_std_e", base["dsnu_std_e"])),
    }


def _pwl_auto_knees(max_in: float, out_max: float, n_segments: int) -> tuple[list[float], list[float]]:
    """Square-root-law PWL: unity slope up to ``a``, then knees on f(x)=a+2sqrt(a)(sqrt(x)-sqrt(a)).

    f is C1 at ``a`` and f(max_in) = out_max, so the decompanded quantisation step
    Delta_in = 1/f'(x) = sqrt(x/a) grows like the shot noise (quantisation/shot variance
    ratio ~ K_hdr / (12 a) in the sqrt region).
    """
    if out_max >= max_in:
        return [0.0, max_in], [0.0, max_in]
    s = math.sqrt(max_in) - math.sqrt(max_in - out_max)
    a = s * s
    xs = np.geomspace(a, max_in, max(2, n_segments))
    ys = a + 2.0 * s * (np.sqrt(xs) - s)
    ys[-1] = out_max
    return [0.0, *xs.tolist()], [0.0, *ys.tolist()]


def build_architecture(hdr_cfg: dict, base: dict) -> HdrArchitecture:
    """Build an :class:`HdrArchitecture` from ``noise.hdr`` and the single-exposure base model.

    ``base`` keys: K_e_per_DN, sigma_d_e, full_well_e, black_DN, bit_depth, t_int_s,
    dark_current_e_per_s, dark_temp_scale, prnu_std, dsnu_std_e, temperature_c.
    """
    arch = str(hdr_cfg.get("architecture", "")).lower()
    if arch not in ARCHITECTURES:
        raise ValueError(f"noise.hdr.architecture must be one of {ARCHITECTURES}, got {arch!r}")
    t0 = float(base["t_int_s"])
    sub = hdr_cfg.get(arch) or {}
    merge = hdr_cfg.get("merge") or {}
    method = str(merge.get("method", "snr_weighted"))
    collectors: list[Collector] = []
    readouts: list[Readout] = []
    notes: dict = {}

    if arch == "dcg":
        fw = float(sub.get("photodiode_full_well_e", base["full_well_e"]))
        collectors.append(Collector("pd", 1.0, fw, t_int_s=t0, **_pd_params(sub, base, t0)))
        readouts.append(_readout("hcg", "pd", sub.get("hcg"), base, full_well=fw))
        readouts.append(_readout("lcg", "pd", sub.get("lcg"), base, full_well=fw))
        mode = str(sub.get("readout", "dual"))
        if mode not in ("dual", "switch"):
            raise ValueError("noise.hdr.dcg.readout must be 'dual' or 'switch'")
        if mode == "switch":
            method = "select"
        notes["dcg_readout"] = mode
    elif arch == "split_pixel":
        for key, default_ratio in (("large", 1.0), ("small", None)):
            d = sub.get(key) or {}
            ratio = float(d.get("sensitivity_ratio", default_ratio if default_ratio is not None else float("nan")))
            if not ratio > 0:
                raise ValueError(f"noise.hdr.split_pixel.{key}.sensitivity_ratio must be > 0")
            rgb = d.get("sensitivity_ratio_rgb")
            fw = float(d.get("full_well_e", base["full_well_e"]))
            name = "lpd" if key == "large" else "spd"
            collectors.append(
                Collector(
                    name,
                    ratio,
                    fw,
                    t_int_s=t0,
                    response_rgb=tuple(float(v) for v in rgb) if rgb else None,
                    **_pd_params(d, base, t0),
                )
            )
            ro = d.get("readouts") or {name: {}}
            for rname, rd in ro.items():
                readouts.append(_readout(f"{name}_{rname}" if rname != name else name, name, rd, base, full_well=fw))
    elif arch == "lofic":
        fw = float(sub.get("photodiode_full_well_e", base["full_well_e"]))
        cap = float(sub.get("capacitor_full_well_e", 0.0))
        if cap <= 0:
            raise ValueError("noise.hdr.lofic.capacitor_full_well_e must be > 0")
        cap_dark = float(sub.get("capacitor_dark_current_e_per_s", 0.0)) * t0 * float(base.get("dark_temp_scale", 1.0))
        collectors.append(
            Collector(
                "pd",
                1.0,
                fw,
                overflow_capacity_e=cap,
                overflow_dark_e=cap_dark,
                t_int_s=t0,
                **_pd_params(sub, base, t0),
            )
        )
        readouts.append(_readout("hcg", "pd", sub.get("hcg"), base, full_well=fw))
        readouts.append(_readout("lcg", "pd", sub.get("lcg"), base, full_well=fw + cap, overflow=True))
    else:
        ratios = [float(v) for v in sub.get("exposure_ratios", [1.0, 1.0 / 16, 1.0 / 256])]
        if not ratios or any(v <= 0 for v in ratios):
            raise ValueError("noise.hdr.multi_exposure.exposure_ratios must be positive")
        gap = float(sub.get("inter_exposure_gap_s", 0.0))
        fw = float(sub.get("photodiode_full_well_e", base["full_well_e"]))
        t_start = 0.0
        pd = _pd_params(sub, base, t0)
        for i, ratio in enumerate(ratios):
            t_i = t0 * ratio
            collectors.append(
                Collector(
                    f"exp{i}",
                    ratio,
                    fw,
                    dark_e=pd["dark_e"] * ratio,
                    prnu_std=pd["prnu_std"],
                    dsnu_std_e=pd["dsnu_std_e"] * ratio,
                    photodiode="pd",
                    t_start_s=t_start,
                    t_int_s=t_i,
                )
            )
            readouts.append(_readout(f"exp{i}", f"exp{i}", sub.get("readout"), base, full_well=fw))
            t_start += t_i + gap
        notes["exposure_order"] = "sequential, in the order listed (t_start_s per capture)"

    if method not in ("snr_weighted", "select"):
        raise ValueError("noise.hdr.merge.method must be 'snr_weighted' or 'select'")
    a = HdrArchitecture(
        name=arch,
        collectors=tuple(collectors),
        readouts=tuple(readouts),
        method=method,
        threshold_fraction=float(merge.get("threshold_fraction", 0.9)),
        hdr_K_e_per_DN=merge.get("hdr_K_e_per_DN"),
        notes=notes,
    )
    if not 0 < a.threshold_fraction <= 1:
        raise ValueError("noise.hdr.merge.threshold_fraction must be in (0, 1]")
    comp = hdr_cfg.get("compand") or {}
    if comp.get("enabled", False):
        bits = int(comp.get("bit_depth", 12))
        out_max = float((1 << bits) - 1) - float(comp.get("black_level_DN", 64.0))
        knees = comp.get("knees", "auto")
        if knees == "auto":
            xin, yout = _pwl_auto_knees(a.max_reference_e / a.K_hdr, out_max, int(comp.get("segments", 12)))
        else:
            xin, yout = [float(k[0]) for k in knees], [float(k[1]) for k in knees]
        if xin[0] != 0 or yout[0] != 0 or np.any(np.diff(xin) <= 0) or np.any(np.diff(yout) <= 0) or yout[-1] > out_max:
            raise ValueError("noise.hdr.compand.knees must start at [0,0], increase strictly and fit the bit depth")
        a = HdrArchitecture(
            **{
                **a.__dict__,
                "compander": Compander(tuple(xin), tuple(yout), bits, float(comp.get("black_level_DN", 64.0))),
            }
        )
    return a


# ---------------------------------------------------------------------------
# Monte Carlo capture simulation
# ---------------------------------------------------------------------------


def _channel_map(shape: tuple[int, ...], bayer_pattern: str | None) -> np.ndarray | None:
    if len(shape) == 3 and shape[-1] == 3:
        return np.broadcast_to(np.arange(3), shape)
    if bayer_pattern and len(shape) == 2:
        ph = _BAYER_PHASE[bayer_pattern.upper()]
        rr, cc = np.indices(shape)
        return np.asarray(ph)[(rr % 2) * 2 + (cc % 2)]
    return None


def response_map(c: Collector, shape: tuple[int, ...], bayer_pattern: str | None = None) -> np.ndarray | float:
    if c.response_rgb is None:
        return c.response
    ch = _channel_map(shape, bayer_pattern)
    if ch is None:
        return c.response
    return np.asarray(c.response_rgb, dtype=np.float64)[ch]


def simulate_captures(
    arch: HdrArchitecture,
    signal_e: np.ndarray,
    rng: np.random.Generator,
    *,
    spatial_rng: np.random.Generator | None = None,
    capture_signal_e: dict[str, np.ndarray] | None = None,
    bayer_pattern: str | None = None,
    use_poisson: bool = True,
) -> dict[str, np.ndarray]:
    """Simulate every readout; returns ``{readout_name: DN}`` (integer-valued float64)."""
    signal_e = np.asarray(signal_e, dtype=np.float64)
    shape = signal_e.shape
    capture_signal_e = capture_signal_e or {}
    spatial_rng = spatial_rng if spatial_rng is not None else np.random.default_rng(0)
    fpn: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in arch.collectors:
        if c.pd not in fpn:
            fpn[c.pd] = (spatial_rng.standard_normal(shape), spatial_rng.standard_normal(shape))
    out: dict[str, np.ndarray] = {}
    for c in arch.collectors:
        if c.name in capture_signal_e:
            sig = np.asarray(capture_signal_e[c.name], dtype=np.float64)
            if sig.shape != shape:
                raise ValueError(f"capture {c.name!r} electrons shape {sig.shape} != {shape}")
        else:
            sig = signal_e * response_map(c, shape, bayer_pattern)
        zp, zd = fpn[c.pd]
        mean = sig * (1.0 + c.prnu_std * zp) + c.dark_e + c.dsnu_std_e * zd
        mean = np.maximum(mean, 0.0)
        q = rng.poisson(mean).astype(np.float64) if use_poisson else mean
        q_pd = np.minimum(q, c.full_well_e)
        q_ov = np.clip(q - c.full_well_e, 0.0, c.overflow_capacity_e) if c.overflow_capacity_e > 0 else None
        d_cap = None
        for r in arch.readouts:
            if r.collector != c.name:
                continue
            charge = q_pd
            if r.reads_overflow:
                if d_cap is None:
                    d_cap = rng.poisson(c.overflow_dark_e, shape).astype(np.float64) if c.overflow_dark_e > 0 else 0.0
                charge = q_pd + (q_ov if q_ov is not None else 0.0) + d_cap
            charge = np.minimum(charge, r.full_well_e)
            e = charge + rng.normal(0.0, r.sigma_e, shape) if r.sigma_e > 0 else charge
            out[r.name] = np.clip(np.rint(e / r.K_e_per_DN + r.black_DN), 0.0, r.max_dn)
    return out


# ---------------------------------------------------------------------------
# Covariance, merge and analytic SNR
# ---------------------------------------------------------------------------


def estimate_covariance(
    arch: HdrArchitecture, mu: np.ndarray, resp: list, *, include_spatial: bool = False
) -> np.ndarray:
    """Covariance (e-^2, reference units) of the per-readout estimates mu_k at signal ``mu``.

    Shape ``mu.shape + (n, n)``.  Temporal: shared shot + dark shot between readouts of the
    same collector, plus read noise and K^2/12 quantisation (EMVA 1288 4.0 Linear).  Spatial
    (``include_spatial``, EMVA 1288 4.0 Linear Sec. 8.5 total SNR): PRNU^2 mu^2 + DSNU^2/r^2,
    fully correlated between collectors on the same photodiode.
    """
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 0.0)
    ro = arch.readouts
    n = len(ro)
    C = np.zeros(mu.shape + (n, n))
    cols = [arch.collector(r.collector) for r in ro]
    for k in range(n):
        for m in range(k, n):
            ck, cm = cols[k], cols[m]
            v = np.zeros_like(mu)
            if ck.name == cm.name:
                shared = ck.dark_e + (ck.overflow_dark_e if ro[k].reads_overflow and ro[m].reads_overflow else 0.0)
                v = v + (resp[k] * mu + shared) / (resp[k] * resp[m])
                if k == m:
                    v = v + (ro[k].sigma_e ** 2 + ro[k].K_e_per_DN ** 2 / 12.0) / resp[k] ** 2
            if include_spatial and ck.pd == cm.pd:
                v = v + ck.prnu_std * cm.prnu_std * mu**2 + ck.dsnu_std_e * cm.dsnu_std_e / (resp[k] * resp[m])
            C[..., k, m] = v
            C[..., m, k] = v
    return C


def _weights(arch: HdrArchitecture, C: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = C.shape[-1]
    eye = np.eye(n, dtype=bool)
    vv = valid[..., :, None] & valid[..., None, :]
    Cm = np.where(vv, C, np.where(eye, 1.0, 0.0))
    if arch.method == "select":
        d = np.where(valid, np.diagonal(C, axis1=-2, axis2=-1), np.inf)
        w = (np.arange(n) == np.argmin(d, axis=-1)[..., None]).astype(np.float64)
    else:
        x = np.linalg.solve(Cm, valid.astype(np.float64)[..., None])[..., 0]
        w = x / np.where(np.sum(x, axis=-1, keepdims=True) == 0, 1.0, np.sum(x, axis=-1, keepdims=True))
    none = ~np.any(valid, axis=-1)
    if np.any(none):
        top = int(np.argmax([t for _, t in [(r.name, dict(arch.transitions_e())[r.name]) for r in arch.readouts]]))
        w[none] = 0.0
        w[none, top] = 1.0
    return w


def merge_captures(
    arch: HdrArchitecture, dn: dict[str, np.ndarray], *, bayer_pattern: str | None = None
) -> tuple[np.ndarray, dict]:
    """Merge per-readout DN into linear HDR reference electrons (dark-subtracted)."""
    ro = arch.readouts
    shape = np.asarray(dn[ro[0].name]).shape
    flat = {r.name: np.asarray(dn[r.name], dtype=np.float64).ravel() for r in ro}
    P = flat[ro[0].name].size
    resp_full = [
        np.broadcast_to(response_map(arch.collector(r.collector), shape, bayer_pattern), shape).ravel() for r in ro
    ]
    thr = [arch.threshold_fraction * arch.saturation_e(r) for r in ro]
    pilot_order = np.argsort([t for t in (dict(arch.transitions_e())[r.name] for r in ro)])
    hdr = np.empty(P)
    n_valid_hist = np.zeros(len(ro) + 1, dtype=np.int64)
    for s in range(0, P, _CHUNK):
        sl = slice(s, min(P, s + _CHUNK))
        charge = np.stack([(flat[r.name][sl] - r.black_DN) * r.K_e_per_DN for r in ro], axis=-1)
        resp = [rf[sl] for rf in resp_full]
        est = np.stack([(charge[:, k] - arch.readout_dark_e(r)) / resp[k] for k, r in enumerate(ro)], axis=-1)
        valid = np.stack([charge[:, k] <= thr[k] for k in range(len(ro))], axis=-1)
        pilot = np.full(est.shape[0], np.nan)
        for k in pilot_order[::-1]:
            pilot = np.where(valid[:, k], est[:, k], pilot)
        pilot = np.where(np.isnan(pilot), est[:, pilot_order[-1]], pilot)
        C = estimate_covariance(arch, np.maximum(pilot, 0.0), resp)
        w = _weights(arch, C, valid)
        hdr[sl] = np.sum(w * est, axis=-1)
        n_valid_hist += np.bincount(valid.sum(axis=-1), minlength=len(ro) + 1)
    info = {"n_valid_readouts_histogram": n_valid_hist.tolist()}
    return hdr.reshape(shape), info


def _slope0(comp: Compander) -> float:
    return comp.knees_out[1] / comp.knees_in[1]


def compand(hdr_dn: np.ndarray, comp: Compander) -> np.ndarray:
    """PWL compand; negative (noise-floor) values continue the first segment below the pedestal."""
    x = np.asarray(hdr_dn, dtype=np.float64)
    y = np.interp(np.clip(x, 0.0, comp.knees_in[-1]), comp.knees_in, comp.knees_out)
    y = np.where(x < 0, x * _slope0(comp), y)
    return np.clip(np.rint(y) + comp.black_DN, 0.0, float((1 << comp.bit_depth) - 1))


def decompand(code: np.ndarray, comp: Compander) -> np.ndarray:
    y = np.asarray(code, dtype=np.float64) - comp.black_DN
    return np.where(y < 0, y / _slope0(comp), np.interp(y, comp.knees_out, comp.knees_in))


def compand_quantisation_var_dn(hdr_dn: np.ndarray, comp: Compander) -> np.ndarray:
    xin, yout = np.asarray(comp.knees_in), np.asarray(comp.knees_out)
    step = np.diff(xin) / np.diff(yout)
    idx = np.clip(np.searchsorted(xin, hdr_dn, side="right") - 1, 0, len(step) - 1)
    return step[idx] ** 2 / 12.0


def theory_snr(
    arch: HdrArchitecture, mu_e: np.ndarray, *, include_spatial: bool = False, include_compander: bool = True
) -> dict[str, np.ndarray]:
    """Analytic SNR(mu) of the merged HDR signal (mu in reference electrons).

    The readout set at each signal follows the merge's transition rule evaluated at the
    mean charge (deterministic switching).  Weights use the temporal covariance (as the
    merge does); the returned variance is ``w^T C w`` with ``C`` optionally including
    DSNU/PRNU (EMVA 1288 4.0 total SNR).  NaN beyond the HDR saturation.
    """
    mu = np.atleast_1d(np.asarray(mu_e, dtype=np.float64))
    ro = arch.readouts
    resp = [arch.collector(r.collector).response for r in ro]
    charge = np.stack([resp[k] * mu + arch.readout_dark_e(r) for k, r in enumerate(ro)], axis=-1)
    valid = np.stack([charge[:, k] <= arch.threshold_fraction * arch.saturation_e(r) for k, r in enumerate(ro)], -1)
    Ct = estimate_covariance(arch, mu, resp)
    w = _weights(arch, Ct, valid)
    C = estimate_covariance(arch, mu, resp, include_spatial=True) if include_spatial else Ct
    var = np.einsum("...i,...ij,...j->...", w, C, w)
    if include_compander and arch.compander is not None:
        var = var + arch.K_hdr**2 * compand_quantisation_var_dn(mu / arch.K_hdr, arch.compander)
    ok = np.any(valid, axis=-1)
    snr = np.where(ok, mu / np.sqrt(var), np.nan)
    return {"mu_e": mu, "snr": snr, "var_e2": np.where(ok, var, np.nan), "weights": w, "n_valid": valid.sum(-1)}


def dynamic_range_db(arch: HdrArchitecture) -> float:
    """20 log10(HDR saturation / temporal noise floor at zero signal)."""
    floor = math.sqrt(float(theory_snr(arch, np.array([0.0]), include_compander=False)["var_e2"][0]))
    return 20.0 * math.log10(arch.max_reference_e / floor)


def describe(arch: HdrArchitecture) -> dict:
    d = {
        "architecture": arch.name,
        "merge_method": arch.method,
        "threshold_fraction": arch.threshold_fraction,
        "K_hdr_e_per_DN": arch.K_hdr,
        "max_reference_e": arch.max_reference_e,
        "dynamic_range_dB": dynamic_range_db(arch),
        "transitions_reference_e": dict(arch.transitions_e()),
        "collectors": [asdict(c) for c in arch.collectors],
        "readouts": [{**asdict(r), "saturation_e": arch.saturation_e(r)} for r in arch.readouts],
        "notes": arch.notes,
    }
    if arch.compander is not None:
        d["compander"] = asdict(arch.compander)
    return json.loads(json.dumps(d, default=lambda o: None if o is None else float(o)).replace("Infinity", "null"))


# ---------------------------------------------------------------------------
# Pipeline hook (apply_emva_noise.py)
# ---------------------------------------------------------------------------


@dataclass
class HdrRun:
    hdr_e: np.ndarray
    dn_noisy: np.ndarray
    dn_clean: np.ndarray
    raw_u16: np.ndarray
    preview_bit_depth: int
    out_dir: Path
    stats: dict


def parse_capture_args(items: list[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for it in items or []:
        name, sep, path = it.partition("=")
        if not sep:
            raise ValueError(f"--hdr-capture-electrons expects NAME=PATH, got {it!r}")
        out[name.strip()] = Path(path.strip())
    return out


def run_hdr(
    cfg: dict,
    base: dict,
    signal_e: np.ndarray,
    *,
    seed: int,
    spatial_seed: int,
    raw_out: Path,
    bayer_pattern: str | None,
    use_poisson: bool,
    capture_paths: dict[str, Path] | None = None,
    capture_loader: Callable[[Path], np.ndarray] | None = None,
    repo: Path | None = None,
) -> HdrRun:
    """Simulate + merge the configured HDR pixel and write ``<raw_out stem>_hdr/`` outputs."""
    hdr_cfg = cfg.get("hdr") or {}
    arch = build_architecture(hdr_cfg, base)
    paths = dict(capture_paths or {})
    cfg_paths = (hdr_cfg.get(arch.name) or {}).get("capture_electrons_npz") or {}
    if isinstance(cfg_paths, list):
        cfg_paths = {c.name: p for c, p in zip(arch.collectors, cfg_paths) if p}
    for k, v in cfg_paths.items():
        p = Path(v)
        paths.setdefault(k, p if p.is_absolute() or repo is None else repo / p)
    unknown = set(paths) - {c.name for c in arch.collectors}
    if unknown:
        raise ValueError(
            f"HDR capture electrons for unknown collectors {sorted(unknown)}; have {[c.name for c in arch.collectors]}"
        )
    if paths and capture_loader is None:
        raise ValueError("capture electrons given but no loader")
    captures = {k: capture_loader(p) for k, p in paths.items()} if paths else {}
    rng = np.random.default_rng([int(seed), 0x48445250])
    spatial_rng = np.random.default_rng([int(spatial_seed), 0x48445250])
    dn = simulate_captures(
        arch,
        signal_e,
        rng,
        spatial_rng=spatial_rng,
        capture_signal_e=captures,
        bayer_pattern=bayer_pattern,
        use_poisson=use_poisson,
    )
    hdr_e, merge_info = merge_captures(arch, dn, bayer_pattern=bayer_pattern)
    hdr_e = np.minimum(hdr_e, arch.max_reference_e)
    K = arch.K_hdr
    hdr_dn = hdr_e / K
    clean_dn = np.clip(np.asarray(signal_e, dtype=np.float64), 0.0, arch.max_reference_e) / K
    lin_bits = max(1, int(math.ceil(math.log2(arch.max_reference_e / K + 1.0))))
    out_dir = raw_out.parent / f"{raw_out.stem}_hdr"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _mono(a: np.ndarray) -> np.ndarray:
        return a if a.ndim == 2 else np.mean(a, axis=2)

    for r in arch.readouts:
        np.rint(_mono(dn[r.name])).astype("<u2").tofile(out_dir / f"{r.name}.raw16")
    payload = {"hdr_e": hdr_e.astype(np.float32), "hdr_dn": hdr_dn.astype(np.float32), "K_hdr_e_per_DN": np.float64(K)}
    if arch.compander is not None:
        codes = compand(hdr_dn, arch.compander)
        payload["companded_code"] = codes.astype(np.uint16)
        raw = np.rint(_mono(codes)).astype(np.uint16)
        qerr = decompand(codes, arch.compander) * K - hdr_e
        merge_info["compand_rms_error_e"] = float(np.sqrt(np.mean(qerr**2)))
    else:
        if lin_bits > 16:
            print(
                f"warning: linear HDR needs {lin_bits} bits; raw16 output is clipped at 65535 DN "
                f"(full-precision linear HDR is in {out_dir / 'hdr_linear.npz'}; enable noise.hdr.compand)",
                file=sys.stderr,
            )
        raw = np.clip(np.rint(_mono(hdr_dn)), 0, 65535).astype(np.uint16)
    np.savez_compressed(out_dir / "hdr_linear.npz", **payload)
    stats = {
        **describe(arch),
        "linear_hdr_bits": lin_bits,
        "outputs": {
            "dir": str(out_dir),
            "per_capture_raw16": [f"{r.name}.raw16" for r in arch.readouts],
            "linear_hdr_npz": "hdr_linear.npz",
            "raw_out_content": "companded_code" if arch.compander is not None else "linear_hdr_dn_clipped_16bit",
        },
        "capture_inputs": {k: str(v) for k, v in paths.items()},
        "hdr_e_mean": float(np.mean(hdr_e)),
        **merge_info,
    }
    (out_dir / "hdr_metadata.json").write_text(json.dumps(stats, indent=2) + "\n")
    return HdrRun(
        hdr_e=hdr_e,
        dn_noisy=hdr_dn,
        dn_clean=clean_dn,
        raw_u16=raw,
        preview_bit_depth=lin_bits,
        out_dir=out_dir,
        stats=stats,
    )


def base_from_noise_config(cfg: dict) -> dict:
    """Single-exposure base parameters from a noise config, as parsed by apply_emva_noise."""
    emva, adc, sensor = cfg.get("emva") or {}, cfg.get("adc") or {}, cfg.get("sensor") or {}
    iso = float(emva.get("iso_gain_factor", 1.0))
    sigma = float(emva.get("sigma_d_e", 2.0))
    amp = float(emva.get("sigma_amp_e", 0.0))
    if amp > 0:
        sigma = math.hypot(sigma, amp * iso)
    t_ref = float(emva.get("dark_current_reference_temp_c", 20.0))
    t_c = float(emva.get("temperature_c", t_ref))
    ea = float(emva.get("dark_activation_energy_eV", 0.0))
    if ea > 0:
        tk, t0k = t_c + 273.15, t_ref + 273.15
        scale = (tk / t0k) ** 1.5 * math.exp((ea / 8.617333262e-5) * (1.0 / t0k - 1.0 / tk))
    else:
        scale = 2.0 ** ((t_c - t_ref) / max(1e-6, float(emva.get("dark_current_doubling_per_c", 6.0))))
    return {
        "K_e_per_DN": float(emva.get("overall_system_gain_K_e_per_DN", 0.08)) / iso,
        "sigma_d_e": sigma,
        "full_well_e": float(adc["full_well_e"]) / iso,
        "black_DN": float(emva.get("black_level_DN", 64.0)),
        "bit_depth": int(adc.get("bit_depth", 12)),
        "t_int_s": float(sensor.get("integration_time_s", 0.01)),
        "dark_current_e_per_s": float(emva.get("dark_current_e_per_s", 0.0)),
        "dark_temp_scale": scale,
        "prnu_std": float(emva.get("prnu_std_fraction", 0.005)),
        "dsnu_std_e": float(emva.get("dsnu_std_e", 0.3)),
        "temperature_c": t_c,
    }


def compander_roundtrip_e(arch: HdrArchitecture, hdr_e: np.ndarray) -> np.ndarray:
    if arch.compander is None:
        return hdr_e
    return decompand(compand(hdr_e / arch.K_hdr, arch.compander), arch.compander) * arch.K_hdr
