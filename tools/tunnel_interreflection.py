"""Interreflected light in a road tunnel tube (radiosity on the extruded cross-section).

The tube is the closed cross-section ``floor + tube_profile()`` extruded along z between the two portals.
It is cut into patches: ~``seg_m`` wide segments of the profile by ``dz_m`` slices of tube length. The
profile is the same at every z, so the form factors depend only on (source segment, target segment,
slice offset) and are estimated once by Monte Carlo cosine-weighted rays from each segment. Each bounce
is then a convolution along z. Light that leaves through a portal is lost; daylight that comes in
through the portals and the cars are not modelled. All surfaces are Lambertian with luminous reflectances.
"""

from __future__ import annotations

import numpy as np


def _subdivide(pts: list[tuple[float, float]], seg_m: float) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / seg_m)))
        out += [(a + (b - a) * i / n, a + (b - a) * (i + 1) / n) for i in range(n)]
    return out


class TubeRadiosity:
    def __init__(
        self,
        profile: list[tuple[float, float]],
        floor_albedo,
        lining_albedo: float,
        length_m: float,
        *,
        seg_m: float = 0.5,
        dz_m: float = 1.0,
        rays: int = 4096,
        seed: int = 0,
    ) -> None:
        """``profile``: wall/vault polyline from the left floor corner up and over to the right floor corner
        (points listed left to right over the top, as ``tube_profile`` gives them). ``floor_albedo(x)``: floor
        reflectance at lateral position x (road vs walkways)."""
        segs = _subdivide([profile[-1], profile[0]], seg_m) + _subdivide(
            profile, seg_m
        )  # floor right->left, then walls
        self.a = np.array([s[0] for s in segs])
        self.b = np.array([s[1] for s in segs])
        t = self.b - self.a
        self.len = np.linalg.norm(t, axis=1)
        self.n = np.stack([t[:, 1], -t[:, 0]], axis=1) / self.len[:, None]  # inward for this winding
        self.mid = 0.5 * (self.a + self.b)
        if self.n[0, 1] < 0:
            self.n = -self.n
        n_floor = len(_subdivide([profile[-1], profile[0]], seg_m))
        self.is_floor = np.arange(len(segs)) < n_floor
        self.rho = np.where(self.is_floor, [floor_albedo(x) for x in self.mid[:, 0]], lining_albedo)
        self.dz, self.nz = dz_m, int(np.ceil(length_m / dz_m))
        self.z = (np.arange(self.nz) + 0.5) * dz_m
        self._form_factors(rays, np.random.default_rng(seed))

    def _hit(self, o: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """First segment hit and 3-D ray parameter for rays o (Nx2 xy) + t d (Nx3)."""
        e = self.b - self.a
        best_t, best_j = np.full(len(o), np.inf), np.full(len(o), -1)
        for j in range(len(e)):
            den = d[:, 0] * e[j, 1] - d[:, 1] * e[j, 0]
            w = self.a[j] - o
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (w[:, 0] * e[j, 1] - w[:, 1] * e[j, 0]) / den
                u = (w[:, 0] * d[:, 1] - w[:, 1] * d[:, 0]) / den
            ok = (t > 1e-6) & (u >= 0) & (u <= 1) & (t < best_t)
            best_t, best_j = np.where(ok, t, best_t), np.where(ok, j, best_j)
        return best_j, best_t

    def _form_factors(self, rays: int, rng: np.random.Generator) -> None:
        S, K = len(self.len), self.nz
        self.F = np.zeros((S, S, 2 * K - 1))  # F[i, j, k + K - 1]: fraction of i's flux reaching j, k slices on
        for i in range(S):
            u, z0 = rng.random(rays), rng.random(rays) * self.dz
            o = self.a[i] + u[:, None] * (self.b[i] - self.a[i])
            r1, r2 = rng.random(rays), rng.random(rays)  # cosine-weighted about the inward normal
            st, phi = np.sqrt(r1), 2 * np.pi * r2
            tang = np.array([-self.n[i, 1], self.n[i, 0]])
            dxy = np.sqrt(1 - r1)[:, None] * self.n[i] + (st * np.cos(phi))[:, None] * tang
            d = np.column_stack([dxy, st * np.sin(phi)])
            j, t = self._hit(o + 1e-9 * self.n[i], d)
            k = np.floor((z0 + t * d[:, 2]) / self.dz).astype(np.int64)
            ok = (j >= 0) & (np.abs(k) < K)
            np.add.at(self.F[i], (j[ok], k[ok] + K - 1), 1.0 / rays)

    def bounce(self, e_direct: np.ndarray, tol: float = 1e-5, max_iter: int = 200) -> np.ndarray:
        """Indirect irradiance (S x nz) from direct irradiance ``e_direct`` (S x nz), all bounces."""
        S, K = e_direct.shape
        n = 1 << int(np.ceil(np.log2(3 * K)))
        Fh = np.fft.rfft(self.F, n=n, axis=2)
        area = self.len * self.dz
        e_ind = np.zeros_like(e_direct)
        src = e_direct
        for _ in range(max_iter):
            flux = np.fft.rfft(self.rho[:, None] * src * area[:, None], n=n, axis=1)
            got = np.fft.irfft(np.einsum("ijf,if->jf", Fh, flux), n=n, axis=1)[:, K - 1 : 2 * K - 1] / area[:, None]
            e_ind += got
            src = got
            if got.max() <= tol * max(e_ind.max(), 1e-30):
                break
        return e_ind

    def direct_from_lamps(self, lamps, lamp_y: float, lamp_size: tuple[float, float]) -> np.ndarray:
        """Irradiance (S x nz) on the patch centres from unit-luminance downward Lambertian lamps (x, z)."""
        w, ln = lamp_size
        sub = np.linspace(-0.5 + 1 / 12, 0.5 - 1 / 12, 6) * ln
        da = w * ln / len(sub)
        px, py = self.mid[:, 0][:, None], self.mid[:, 1][:, None]
        nx, ny = self.n[:, 0][:, None], self.n[:, 1][:, None]
        e = np.zeros((len(self.len), self.nz))
        for x, z in lamps:
            for dz in sub:
                vx, vy, vz = x - px, lamp_y - py, (z + dz) - self.z[None, :]
                d2 = vx * vx + vy * vy + vz * vz
                d = np.sqrt(d2)
                cl = np.clip(vy / d, 0.0, None)  # lamp emits downward (normal -y)
                cp = np.clip((nx * vx + ny * vy) / d, 0.0, None)
                e += da * cl * cp / d2
        return e

    def floor_mean(self, e: np.ndarray, x0: float, x1: float, z0: float, z1: float) -> float:
        rows = self.is_floor & (self.mid[:, 0] >= x0) & (self.mid[:, 0] <= x1)
        cols = (self.z >= z0) & (self.z <= z1)
        return float(e[np.ix_(rows, cols)].mean())
