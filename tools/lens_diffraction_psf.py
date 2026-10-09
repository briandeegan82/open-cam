"""Field- and wavelength-dependent diffraction PSFs of a traced lens prescription.

pbrt-v4's ``realistic`` camera traces a lens prescription (``config/lenses/*.dat``) with
geometric rays only, so its images contain the lens's geometric aberrations, defocus and
vignetting but no diffraction. This module computes the wave-optics PSF of the same
prescription and applies it, spatially varying and per spectral bucket, to a rendered
SpectralFilm EXR before the radiance -> electrons step.

Physics (scalar diffraction, Debye approximation)
--------------------------------------------------
1. Rays from an object point are aimed (2-D Newton iteration) at a regular grid on the
   aperture stop and traced sequentially through every spherical interface, accumulating the
   optical path length (OPL). The stop and every element clip rays exactly like pbrt, so the
   pupil of an off-axis point is the real vignetted ("cat's-eye") pupil. Its boundary is found
   by bisection along stop radii.
2. The chief ray (stop centre) fixes the image point P' on the film; the reference sphere is
   centred on P' and passes through the chief ray's intersection with the paraxial exit-pupil
   plane. The wavefront aberration is W = OPL(Q) - OPL(chief) at each ray's intersection Q
   with that sphere (Welford 1986, ch. 6; Born & Wolf 1999, sec. 9.1).
3. The field near focus is the Debye integral over the direction cosines s = (P' - Q)/R of the
   converging wave (Born & Wolf 1999, sec. 8.8; Goodman 2005, sec. 6.4 for the generalized pupil
   P exp(ikW)):  U(rho) = \\int a(s) exp(ik W(s)) exp(ik s.rho) dOmega,  rho in the film plane.
   With s sampled on a regular grid this is a 2-D FFT (Goodman 2005, ch. 4-6); PSF = |U|^2.
   Using direction cosines (not exit-pupil coordinates) keeps oblique and wide-angle fields
   right: the film-plane projection and pupil distortion are in the ray data.
   The pupil amplitude |a|^2 = dP/dOmega' follows from ray density (power per ray tube for a
   Lambertian object, cos(theta) dOmega_obj, over its image-side solid angle), i.e. the pupil is
   apodised by pupil aberration as a real lens is. ``pupil_amplitude: uniform`` switches it off.
   An optional Gaussian apodiser and an N-blade polygonal stop are supported.
4. The PSF is sampled on a fine grid (spacing <= lambda N_w / 2, so |U|^2 is unaliased) and
   converted to a kernel at the sensor pixel pitch by keeping the OTF inside the pixel Nyquist
   band (Fourier cropping): the digital kernel's frequency response equals the lens OTF for
   |f| < 1/(2p). The render already contains the pixel aperture (pbrt pixel filter), so the
   kernel must not integrate over the pixel a second time.
5. Spatial variation: the film is divided into a grid of tiles; each tile centre gets the exact
   traced PSF (rotated to its azimuth; a rotationally symmetric lens is assumed, as pbrt's
   lens format is). Each source pixel is spread by the bilinear interpolation of the PSFs of the
   four nearest tile centres, out = sum_k h_k * (w_k f), which conserves energy exactly
   (piecewise / interpolated spatially variant blur, Nagy & O'Leary 1998).

Combining with pbrt's geometric aberrations (no double counting)
-----------------------------------------------------------------
``combine: diffraction_only`` (for ``realistic`` renders): the kernel is the PSF of the traced,
vignetted pupil with W = 0. pbrt already convolved the scene with the geometric PSF, so the
system MTF becomes MTF_geom x MTF_diff. This is exact when W << lambda (diffraction limited)
and when W >> lambda (geometric limit), and an approximation in between: the true wave PSF
|FT(P exp(ikW))|^2 is not a convolution of the two (Hopkins 1955 for defocus). It is the only
choice that keeps pbrt's depth-dependent defocus, occlusion and per-pixel vignetting.
``combine: wave_optics`` (for ``pinhole``/``perspective`` renders): the full PSF with W is
applied to a geometrically perfect image, giving the exact wave-optics PSF at one object
distance (``object_distance_m``), but no depth-dependent blur and no distortion.

Not modelled: dispersion (pbrt lens files have one index per glass, so there is no chromatic
aberration; the PSF still varies with wavelength through lambda N and W/lambda), polarisation
and vector (high-NA) effects, Fresnel losses, coatings, scatter, multiple reflections (ghosts),
sensor microlens / crosstalk, and diffraction at element edges other than the pupil boundary.

References
----------
J. W. Goodman, *Introduction to Fourier Optics*, 3rd ed., Roberts & Co., 2005 (ch. 6).
M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge Univ. Press, 1999 (8.5, 8.8, 9.1).
W. T. Welford, *Aberrations of Optical Systems*, Adam Hilger, 1986 (OPD, Seidel sums).
H. H. Hopkins, "The frequency response of a defocused optical system," Proc. R. Soc. Lond. A
231, 91-103 (1955).
J. G. Nagy and D. P. O'Leary, "Restoring images degraded by spatially variant blur," SIAM J.
Sci. Comput. 19(4), 1063-1082 (1998).
M. Pharr, W. Jakob, G. Humphreys, *Physically Based Rendering*, 4th ed., 2023 (sec. 5.3,
RealisticCamera: lens format, stop clamp, thick-lens focusing replicated here).
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lens_prescription import load_lens_file

# pbrt RGB film channels: same representative wavelengths as apply_spectral_psf.
RGB_CENTER_NM = {"R": 620.0, "G": 540.0, "B": 460.0}
PBRT_DEFAULT_FILM_DIAGONAL_MM = 35.0


# =====================================================================
# Lens system and sequential ray trace (mm; z = 0 at the front vertex, +z toward the film)
# =====================================================================
@dataclass(frozen=True)
class LensSystem:
    radius: np.ndarray
    thickness: np.ndarray
    n_after: np.ndarray
    semi_aperture: np.ndarray
    stop_index: int
    aperture_blades: int = 0
    blade_rotation_deg: float = 0.0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        aperture_diameter_mm: float | None = None,
        *,
        aperture_blades: int = 0,
        blade_rotation_deg: float = 0.0,
    ) -> LensSystem:
        rows = np.asarray(load_lens_file(Path(path)), dtype=np.float64)
        radius, thick, eta, ap = rows.T.copy()
        stop = int(np.flatnonzero(radius == 0.0)[0])
        if aperture_diameter_mm is not None:
            if aperture_diameter_mm <= 0:
                raise ValueError("aperture diameter must be positive")
            ap[stop] = min(float(aperture_diameter_mm), ap[stop])  # pbrt clamps to the listed stop
        n_after = np.where(eta == 0.0, 1.0, eta)
        return cls(radius, thick, n_after, 0.5 * ap, stop, int(aperture_blades), float(blade_rotation_deg))

    @property
    def n_surfaces(self) -> int:
        return int(self.radius.size)

    @property
    def z_vertex(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.thickness[:-1])])

    @property
    def stop_semi_diameter(self) -> float:
        return float(self.semi_aperture[self.stop_index])

    def n_before(self, i: int) -> float:
        return 1.0 if i == 0 else float(self.n_after[i - 1])

    def inside_stop(self, x: np.ndarray, y: np.ndarray, margin: float = 0.0) -> np.ndarray:
        r = self.stop_semi_diameter * (1.0 + margin)
        if self.aperture_blades < 3:
            return x * x + y * y <= r * r
        n = self.aperture_blades
        th = np.arctan2(y, x) - math.radians(self.blade_rotation_deg)
        # Regular N-gon inscribed in the stop circle: boundary radius r cos(pi/n)/cos(theta_mod).
        th_mod = np.mod(th, 2 * math.pi / n) - math.pi / n
        return np.hypot(x, y) * np.cos(th_mod) <= r * math.cos(math.pi / n)


def _refract(d, nrm, n1, n2):
    cos_dot = np.sum(nrm * d, axis=1)
    nrm = np.where((cos_dot > 0)[:, None], -nrm, nrm)
    cos_i = -np.sum(nrm * d, axis=1)
    mu = n1 / n2
    k = 1.0 - mu * mu * (1.0 - cos_i * cos_i)
    ok = k >= 0.0
    d2 = mu * d + (mu * cos_i - np.sqrt(np.clip(k, 0.0, None)))[:, None] * nrm
    return d2 / np.linalg.norm(d2, axis=1, keepdims=True), ok


def trace(
    lens: LensSystem,
    origins: np.ndarray,
    dirs: np.ndarray,
    *,
    surfaces: range | None = None,
    reverse: bool = False,
    clip: bool = True,
    z_shift: float = 0.0,
) -> dict:
    """Sequential real-ray trace; returns final position/direction, OPL, validity, stop hit."""
    o = np.array(origins, dtype=np.float64, ndmin=2)
    d = np.array(dirs, dtype=np.float64, ndmin=2)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    n_rays = o.shape[0]
    ok = np.isfinite(o).all(axis=1) & np.isfinite(d).all(axis=1)
    opl = np.zeros(n_rays)
    stop_xy = np.full((n_rays, 2), np.nan)
    zv_all = lens.z_vertex + z_shift
    order = surfaces if surfaces is not None else range(lens.n_surfaces)
    if reverse:
        order = reversed(list(order))
    for i in order:
        zv, rad = zv_all[i], lens.radius[i]
        if rad == 0.0:
            t = (zv - o[:, 2]) / d[:, 2]
            nrm = None
        else:
            c = np.array([0.0, 0.0, zv + rad])
            oc = o - c
            b = np.sum(oc * d, axis=1)
            disc = b * b - (np.sum(oc * oc, axis=1) - rad * rad)
            ok &= disc >= 0.0
            sq = np.sqrt(np.clip(disc, 0.0, None))
            t1, t2 = -b - sq, -b + sq
            z1, z2 = o[:, 2] + t1 * d[:, 2], o[:, 2] + t2 * d[:, 2]
            t = np.where(np.abs(z1 - zv) <= np.abs(z2 - zv), t1, t2)
        p = o + t[:, None] * d
        n_in = lens.n_after[i] if reverse else lens.n_before(i)
        opl = opl + n_in * t
        if rad == 0.0:
            stop_xy = p[:, :2].copy()
            if clip:
                ok &= lens.inside_stop(p[:, 0], p[:, 1])
        else:
            if clip:
                ok &= p[:, 0] ** 2 + p[:, 1] ** 2 <= lens.semi_aperture[i] ** 2
            nrm = (p - np.array([0.0, 0.0, zv + rad])) / rad
            n_out = lens.n_before(i) if reverse else lens.n_after[i]
            d, ok_r = _refract(d, nrm, n_in, n_out)
            ok &= ok_r
        o = p
    return {"o": o, "d": d, "opl": opl, "ok": ok, "stop_xy": stop_xy}


# =====================================================================
# Paraxial helpers and pbrt-compatible focusing
# =====================================================================
def _paraxial_front_to_stop(lens: LensSystem) -> tuple[float, float]:
    """(A, B): paraxial y at the stop for (y0=1, u0=0) and (y0=0, u0=1) at the front vertex."""
    out = []
    for y, u in ((1.0, 0.0), (0.0, 1.0)):
        n = 1.0
        for i in range(lens.stop_index):
            n2 = lens.n_after[i]
            phi = (n2 - n) / lens.radius[i]
            u = (n * u - y * phi) / n2
            n = n2
            y += lens.thickness[i] * u
        out.append(y)
    return out[0], out[1]


def entrance_pupil(lens: LensSystem) -> tuple[float, float]:
    """Paraxial entrance pupil (z position, magnification stop -> pupil)."""
    a, b = _paraxial_front_to_stop(lens)
    return b / a, 1.0 / a


def exit_pupil_z(lens: LensSystem) -> float:
    """Paraxial exit-pupil z: image of the stop centre through the rear group."""
    y, u, n = 0.0, 1e-3, lens.n_after[lens.stop_index]
    y += lens.thickness[lens.stop_index] * u
    for i in range(lens.stop_index + 1, lens.n_surfaces):
        n2 = lens.n_after[i]
        phi = (n2 - n) / lens.radius[i]
        u = (n * u - y * phi) / n2
        n = n2
        if i < lens.n_surfaces - 1:
            y += lens.thickness[i] * u
    return float(lens.z_vertex[-1] - y / u)


def pbrt_film_distance_mm(lens: LensSystem, focus_distance_mm: float, film_diagonal_mm: float) -> float:
    """Rear vertex -> film distance chosen by pbrt-v4 ``RealisticCamera::FocusThickLens``.

    Replicates pbrt (cameras.cpp): thick-lens cardinal points from two real rays at height
    0.001 x film diagonal, then the lens shift that images ``focus_distance`` (measured from the
    film, pbrt's camera origin) onto the film.
    """
    t_last0 = float(lens.thickness[-1])
    z_film0 = float(lens.z_vertex[-1]) + t_last0
    x = 0.001 * film_diagonal_mm
    front_z_cam = float(np.sum(lens.thickness))  # pbrt LensFrontZ (camera space)

    def to_my(o_cam, d_cam):
        return np.array([[o_cam[0], o_cam[1], z_film0 - o_cam[2]]]), np.array([[d_cam[0], d_cam[1], -d_cam[2]]])

    def to_cam(o_my, d_my):
        return np.array([o_my[0, 0], o_my[0, 1], z_film0 - o_my[0, 2]]), np.array([d_my[0, 0], d_my[0, 1], -d_my[0, 2]])

    def cardinal(o_in, o_out, d_out):
        tf = -o_out[0] / d_out[0]
        fz = -(o_out[2] + tf * d_out[2])
        tp = (o_in[0] - o_out[0]) / d_out[0]
        pz = -(o_out[2] + tp * d_out[2])
        return pz, fz

    o_s = np.array([x, 0.0, front_z_cam + 1000.0])
    om, dm = to_my(o_s, np.array([0.0, 0.0, -1.0]))
    r = trace(lens, om, dm, clip=False)
    o_f, d_f = to_cam(r["o"], r["d"])
    pz0, fz0 = cardinal(o_s, o_f, d_f)
    o_s2 = np.array([x, 0.0, t_last0 - 1000.0])
    om, dm = to_my(o_s2, np.array([0.0, 0.0, 1.0]))
    r = trace(lens, om, dm, clip=False, reverse=True)
    o_f, d_f = to_cam(r["o"], r["d"])
    pz1, _fz1 = cardinal(o_s2, o_f, d_f)
    f = fz0 - pz0
    z = -focus_distance_mm
    c = (pz1 - z - pz0) * (pz1 - z - 4 * f - pz0)
    if c <= 0:
        raise ValueError(f"focus distance {focus_distance_mm} mm too short for this lens")
    delta = 0.5 * (pz1 - z + pz0 - math.sqrt(c))
    return t_last0 + delta


# =====================================================================
# Ray aiming and field pupils
# =====================================================================
def aim_rays(lens: LensSystem, obj: np.ndarray, stop_targets: np.ndarray, iters: int = 12) -> np.ndarray:
    """Directions from object point ``obj`` whose rays hit ``stop_targets`` (M x 2 mm) on the stop."""
    z_ep, m_ep = entrance_pupil(lens)
    tgt = np.asarray(stop_targets, dtype=np.float64)
    aim = np.column_stack([tgt * m_ep, np.full(len(tgt), z_ep)])
    dz = aim[:, 2] - obj[2]
    ab = (aim[:, :2] - obj[None, :2]) / dz[:, None]
    surf = range(lens.stop_index + 1)
    o = np.repeat(obj[None, :], len(tgt), axis=0)

    def hit(ab_):
        r = trace(lens, o, np.column_stack([ab_, np.ones(len(ab_))]), surfaces=surf, clip=False)
        return np.where(r["ok"][:, None], r["stop_xy"], np.nan)

    scale = 1.0 / max(abs(obj[2] - z_ep), 1.0)
    for _ in range(iters):
        f0 = hit(ab) - tgt
        if np.nanmax(np.abs(f0), initial=0.0) < 1e-10:
            break
        h = 1e-4 * scale
        ja = (hit(ab + [h, 0.0]) - tgt - f0) / h
        jb = (hit(ab + [0.0, h]) - tgt - f0) / h
        det = ja[:, 0] * jb[:, 1] - jb[:, 0] * ja[:, 1]
        da = (jb[:, 1] * f0[:, 0] - jb[:, 0] * f0[:, 1]) / det
        db = (-ja[:, 1] * f0[:, 0] + ja[:, 0] * f0[:, 1]) / det
        step = np.column_stack([da, db])
        step = np.where(np.isfinite(step), step, 0.0)
        ab = ab - step
    res = np.abs(hit(ab) - tgt).max(axis=1)
    dirs = np.column_stack([ab, np.ones(len(ab))])
    dirs[~(res < 1e-7)] = np.nan
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


@dataclass
class FieldPupil:
    """Traced pupil of one object point: scattered samples in image-space direction cosines."""

    image_point: np.ndarray  # P' on the film (mm)
    image_height_mm: float  # |P'_xy|, image at (0, -h)
    object_height_mm: float
    s: np.ndarray  # (M, 2) direction cosines relative to the chief ray's
    W_mm: np.ndarray  # (M,) wavefront aberration on the reference sphere
    amp: np.ndarray  # (M,) radiometric pupil amplitude (ray density)
    stop_uv: np.ndarray  # (M, 2) normalised stop coordinates
    spot_mm: np.ndarray  # (n, 2) film intercepts - P' of the interior grid rays
    spot_weight: np.ndarray  # (n,) power per ray
    sz_chief: float
    pupil_area_rel_stop: float  # interior valid ray count / unvignetted count
    _interp: object = field(default=None, repr=False)

    @property
    def s_extent(self) -> float:
        """Diameter-equivalent extent 2 max|s| (sets the fine sampling)."""
        return 2.0 * float(np.max(np.hypot(self.s[:, 0], self.s[:, 1])))

    @property
    def working_f_number(self) -> float:
        return 1.0 / self.s_extent

    def interpolator(self):
        if self._interp is None:
            from scipy.interpolate import LinearNDInterpolator

            self._interp = LinearNDInterpolator(self.s, np.column_stack([self.W_mm, self.amp]), fill_value=np.nan)
        return self._interp


def _nearest_fill(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    if valid.all() or not valid.any():
        return a
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def _jacobian_abs(fx: np.ndarray, fy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    fxr, fxc = np.gradient(fx)
    fyr, fyc = np.gradient(fy)
    j = np.abs(fxc * fyr - fxr * fyc)
    good = valid & np.isfinite(j)
    return _nearest_fill(np.where(good, j, np.nan), good)


def object_height_for_image_height(
    lens: LensSystem, z_obj: float, z_film: float, image_height_mm: float, *, tol: float = 1e-7
) -> float:
    """Object height (+y) whose chief ray lands at film height -image_height_mm."""
    if image_height_mm <= 0:
        return 0.0

    def landing(y0):
        obj = np.array([0.0, y0, z_obj])
        d = aim_rays(lens, obj, np.zeros((1, 2)))
        r = trace(lens, obj[None, :], d, clip=False)
        t = (z_film - r["o"][0, 2]) / r["d"][0, 2]
        return -(r["o"][0, 1] + t * r["d"][0, 1])

    m0 = landing(1e-3) / 1e-3
    y = image_height_mm / m0
    for _ in range(30):
        h = landing(y)
        if abs(h - image_height_mm) < tol:
            break
        dh = (landing(y * (1 + 1e-6)) - h) / (y * 1e-6)
        y -= (h - image_height_mm) / dh
    if not math.isfinite(y):
        raise ValueError(f"chief ray cannot reach image height {image_height_mm} mm")
    return float(y)


def trace_field_pupil(
    lens: LensSystem,
    z_obj: float,
    z_film: float,
    image_height_mm: float,
    *,
    n_pupil: int = 97,
    n_boundary: int = 256,
    pupil_amplitude: str = "radiometric",
    apodisation_w: float | None = None,
) -> FieldPupil:
    """Trace the pupil of the object point imaged at film (0, -image_height_mm)."""
    y0 = object_height_for_image_height(lens, z_obj, z_film, image_height_mm)
    obj = np.array([0.0, y0, z_obj])
    rs = lens.stop_semi_diameter
    g = np.linspace(-1.0, 1.0, n_pupil)
    uu, vv = np.meshgrid(g, g)  # rows = v (stop y), cols = u (stop x)
    in_stop = lens.inside_stop(uu * rs, vv * rs)
    dirs = aim_rays(lens, obj, np.column_stack([uu.ravel(), vv.ravel()]) * rs)
    o = np.repeat(obj[None, :], dirs.shape[0], axis=0)
    r = trace(lens, o, dirs)
    valid = (r["ok"] & np.isfinite(dirs).all(axis=1)).reshape(uu.shape) & in_stop

    chief_d = aim_rays(lens, obj, np.zeros((1, 2)))
    rc = trace(lens, obj[None, :], chief_d, clip=False)
    tc = (z_film - rc["o"][0, 2]) / rc["d"][0, 2]
    p_img = rc["o"][0] + tc * rc["d"][0]
    z_xp = exit_pupil_z(lens)
    c_xp = rc["o"][0] + (z_xp - rc["o"][0, 2]) / rc["d"][0, 2] * rc["d"][0]
    radius_ref = float(np.linalg.norm(p_img - c_xp))
    opl_chief_sphere = None

    def to_sphere(res):
        q, d = res["o"], res["d"]
        oc = q - p_img
        b = np.sum(oc * d, axis=1)
        disc = b * b - (np.sum(oc * oc, axis=1) - radius_ref**2)
        t = -b - np.sqrt(np.clip(disc, 0.0, None))
        qs = q + t[:, None] * d
        s = (p_img[None, :] - qs) / radius_ref
        tf = (z_film - q[:, 2]) / d[:, 2]
        film = q[:, :2] + tf[:, None] * d[:, :2] - p_img[None, :2]
        return s, res["opl"] + t, film

    s_c, opl_c, _ = to_sphere(rc)
    opl_chief_sphere = float(opl_c[0])
    s0 = s_c[0, :2]
    s_all, opl_all, film_all = to_sphere(r)
    sx = s_all[:, 0].reshape(uu.shape)
    sy = s_all[:, 1].reshape(uu.shape)
    sz = s_all[:, 2].reshape(uu.shape)
    W = (opl_all - opl_chief_sphere).reshape(uu.shape)

    # Radiometric amplitude |a|^2 = dP/dOmega' with dP ~ |d(l_o, m_o)/d(u, v)| (Lambertian object).
    if pupil_amplitude == "radiometric":
        dvx = dirs[:, 0].reshape(uu.shape)
        dvy = dirs[:, 1].reshape(uu.shape)
        j_obj = _jacobian_abs(np.where(valid, dvx, np.nan), np.where(valid, dvy, np.nan), valid)
        j_img = _jacobian_abs(np.where(valid, sx, np.nan), np.where(valid, sy, np.nan), valid)
        power = j_obj
        amp2 = j_obj / (j_img * sz)
    elif pupil_amplitude == "uniform":
        power = np.ones_like(sx)
        amp2 = np.ones_like(sx)
    else:
        raise ValueError('pupil_amplitude must be "radiometric" or "uniform"')

    # Exact pupil boundary: bisection along stop radii.
    ang = np.linspace(0.0, 2 * math.pi, n_boundary, endpoint=False)
    cu, cv = np.cos(ang), np.sin(ang)
    lo, hi = np.zeros(n_boundary), np.full(n_boundary, 1.0 + 1e-9)
    if lens.aperture_blades >= 3:
        n = lens.aperture_blades
        th_mod = np.mod(ang - math.radians(lens.blade_rotation_deg), 2 * math.pi / n) - math.pi / n
        hi = math.cos(math.pi / n) / np.cos(th_mod) * (1.0 - 1e-9)

    def ok_at(rho):
        tg = np.column_stack([rho * cu, rho * cv]) * rs
        dd = aim_rays(lens, obj, tg)
        rr = trace(lens, np.repeat(obj[None, :], n_boundary, axis=0), dd)
        return rr["ok"] & np.isfinite(dd).all(axis=1), rr, dd

    ok_hi, _, _ = ok_at(hi)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        ok_mid, _, _ = ok_at(mid)
        lo = np.where(ok_mid | ok_hi, np.where(ok_hi, hi, mid), lo)
        hi = np.where(ok_hi, hi, np.where(ok_mid, hi, mid))
        if ok_hi.all():
            break
    ok_b, rb, db = ok_at(lo)
    sb, oplb, _ = to_sphere(rb)
    keep_b = ok_b & (lo > 0)
    # Boundary amplitudes: nearest interior sample.
    flat_valid = np.flatnonzero(valid.ravel())
    pts_uv = np.column_stack([uu.ravel()[flat_valid], vv.ravel()[flat_valid]])
    buv = np.column_stack([lo * cu, lo * cv])
    from scipy.spatial import cKDTree

    nn = cKDTree(pts_uv).query(buv)[1]
    amp_int = np.sqrt(np.clip(amp2.ravel()[flat_valid], 0.0, None))
    if apodisation_w is not None:
        rho2 = uu.ravel()[flat_valid] ** 2 + vv.ravel()[flat_valid] ** 2
        amp_int = amp_int * np.exp(-rho2 / float(apodisation_w) ** 2)  # T = exp(-2 rho^2 / w^2)
    amp_b = np.sqrt(np.clip(amp2.ravel()[flat_valid][nn], 0.0, None))
    if apodisation_w is not None:
        amp_b = amp_b * np.exp(-(lo**2) / float(apodisation_w) ** 2)

    s_pts = np.vstack([np.column_stack([sx.ravel(), sy.ravel()])[flat_valid], sb[keep_b, :2]]) - s0
    w_pts = np.concatenate([W.ravel()[flat_valid], (oplb - opl_chief_sphere)[keep_b]])
    a_pts = np.concatenate([amp_int, amp_b[keep_b]])
    a_pts = a_pts / np.max(a_pts)
    spot = film_all.reshape(*uu.shape, 2)[valid]
    pw = power[valid]
    if apodisation_w is not None:
        pw = pw * np.exp(-2 * (uu[valid] ** 2 + vv[valid] ** 2) / float(apodisation_w) ** 2)
    return FieldPupil(
        image_point=p_img,
        image_height_mm=float(image_height_mm),
        object_height_mm=y0,
        s=s_pts,
        W_mm=w_pts,
        amp=a_pts,
        stop_uv=np.vstack([pts_uv, buv[keep_b]]),
        spot_mm=spot,
        spot_weight=pw / pw.sum(),
        sz_chief=float(s_c[0, 2]),
        pupil_area_rel_stop=float(valid.sum() / max(1, in_stop.sum())),
    )


# =====================================================================
# PSF / OTF / pixel kernels
# =====================================================================
def _rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def pupil_on_grid(fp: FieldPupil, ds: float, nfft: int, *, aberrations: bool, lam_mm: float, m=None) -> np.ndarray:
    """Complex pupil sampled at s = m @ sigma, sigma on an nfft^2 grid with spacing ds (centred)."""
    m = np.eye(2) if m is None else np.asarray(m)
    idx = np.arange(nfft) - nfft // 2
    half = int(math.ceil(0.5 * fp.s_extent / ds)) + 1
    sel = idx[np.abs(idx) <= half]
    sc, sr = np.meshgrid(sel * ds, sel * ds)  # sigma_x (cols), sigma_y (rows)
    sq = m @ np.vstack([sc.ravel(), sr.ravel()])
    vals = fp.interpolator()(sq.T)
    w, a = vals[:, 0], vals[:, 1]
    inside = np.isfinite(w)
    phase = 2 * math.pi * np.where(inside, w, 0.0) / lam_mm if aberrations else 0.0
    sub = np.where(inside, np.where(inside, a, 0.0) * np.exp(1j * phase), 0.0).reshape(sc.shape)
    out = np.zeros((nfft, nfft), dtype=np.complex128)
    i0 = nfft // 2 + sel[0]
    out[i0 : i0 + sel.size, i0 : i0 + sel.size] = sub
    return out


def fine_psf(fp: FieldPupil, lam_nm: float, dx_um: float, nfft: int, *, aberrations: bool = True, m=None) -> np.ndarray:
    """PSF on an nfft^2 film grid of spacing dx (um), centred on the chief ray, unit sum.

    Array axis 0 is +rho_y, axis 1 is +rho_x of the (possibly rotated) frame ``s = m @ sigma``.
    """
    lam_mm = lam_nm * 1e-6
    ds = lam_mm / (nfft * dx_um * 1e-3)
    if fp.s_extent / ds > nfft / 2 + 2:
        raise ValueError("fine grid too coarse for this pupil (|U|^2 would alias); reduce dx")
    p = pupil_on_grid(fp, ds, nfft, aberrations=aberrations, lam_mm=lam_mm, m=m)
    u = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(p)))
    psf = np.abs(u) ** 2
    return psf / psf.sum()


def oversampling_for(fp: FieldPupil, lam_nm: float, pitch_um: float) -> int:
    """Fine samples per pixel so that dx <= lambda / (2 s_extent) (unaliased |U|^2)."""
    return max(1, int(math.ceil(2.0 * fp.s_extent * pitch_um / (lam_nm * 1e-3) * 1.05)))


def auto_kernel_size(fp: FieldPupil, lam_nm: float, pitch_um: float, aberrations: bool) -> int:
    lam_um = lam_nm * 1e-3
    radius = 6.0 * lam_um / fp.s_extent
    if aberrations and fp.spot_mm.size:
        radius += 1e3 * float(np.max(np.hypot(fp.spot_mm[:, 0], fp.spot_mm[:, 1])))
    return max(5, 2 * int(math.ceil(radius / pitch_um)) + 1)


def pixel_kernel(
    fp: FieldPupil,
    lam_nm: float,
    pitch_um: float,
    kernel_size: int,
    *,
    aberrations: bool = True,
    m=None,
    min_pupil_samples: int = 64,
) -> np.ndarray:
    """K x K kernel at the pixel pitch whose DFT equals the lens OTF at frequencies m / (K p).

    The kernel's DFT frequencies (|f| <= 1/(2p), the pixel Nyquist band) are sampled from the
    OTF of a fine, unaliased PSF; out-of-band OTF is discarded (the pbrt render already contains
    the pixel aperture). Sampling the OTF at spacing 1/(K p) is equivalent to wrapping the PSF tail
    beyond the K x K window back into it, which conserves energy and the low-frequency MTF
    (truncation would not). The pupil gets >= ``min_pupil_samples`` across its diameter by
    computing the PSF over a z*K pixel window (pupil sampling = window * s_extent / lambda).
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    os_ = oversampling_for(fp, lam_nm, pitch_um)
    z = max(1, math.ceil(min_pupil_samples * lam_nm * 1e-3 / (kernel_size * pitch_um * fp.s_extent)))
    z += 1 - z % 2  # odd, so the coarse frequencies are a subset of the fine ones
    nfft = kernel_size * z * os_
    psf = fine_psf(fp, lam_nm, pitch_um / os_, nfft, aberrations=aberrations, m=m)
    otf = np.fft.fft2(np.fft.ifftshift(psf))
    h = kernel_size // 2
    idx = z * np.r_[0 : h + 1, -h:0] % nfft
    k = np.real(np.fft.fftshift(np.fft.ifft2(otf[np.ix_(idx, idx)])))
    return k / k.sum()


def mtf_from_psf(psf: np.ndarray, dx_um: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frequency cy/mm, MTF along axis-1 (x) direction, MTF along axis-0 (y) direction)."""
    otf = np.abs(np.fft.fft2(np.fft.ifftshift(psf)))
    n = psf.shape[1]
    f = np.fft.fftfreq(n, d=dx_um * 1e-3)
    pos = slice(0, n // 2)
    return f[pos], otf[0, pos] / otf[0, 0], otf[pos, 0] / otf[0, 0]


def geometric_otf(fp: FieldPupil, freqs_cy_mm: np.ndarray, angle_rad: float) -> np.ndarray:
    """|OTF| of the geometric spot along direction ``angle_rad`` (film frame) - ray-optics limit."""
    proj = fp.spot_mm[:, 0] * math.cos(angle_rad) + fp.spot_mm[:, 1] * math.sin(angle_rad)
    ph = np.exp(-2j * math.pi * np.outer(freqs_cy_mm, proj))
    return np.abs(ph @ fp.spot_weight)


def airy_psf(r_um: np.ndarray, lam_nm: float, f_number: float) -> np.ndarray:
    """Normalised Airy intensity (2 J1(v)/v)^2, v = pi r / (lambda N) (Born & Wolf 8.5.2)."""
    from scipy.special import j1

    v = np.pi * np.asarray(r_um, dtype=np.float64) / (lam_nm * 1e-3 * f_number)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(v == 0, 1.0, (2 * j1(v) / v) ** 2)
    return out


def diffraction_mtf(freq_cy_mm: np.ndarray, lam_nm: float, f_number: float) -> np.ndarray:
    """Circular-pupil incoherent MTF (Goodman 2005, eq. 6-32)."""
    nu = np.clip(np.asarray(freq_cy_mm) * lam_nm * 1e-6 * f_number, 0.0, 1.0)
    return (2 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1 - nu * nu))


# =====================================================================
# Lens model + spatially varying application
# =====================================================================
@dataclass
class LensPsfModel:
    lens: LensSystem
    z_film: float
    z_obj: float
    pitch_um: float
    film_res: tuple[int, int]  # full film (W, H) in pixels
    combine: str = "diffraction_only"
    tile_grid: tuple[int, int] = (9, 7)
    n_pupil: int = 97
    pupil_amplitude: str = "radiometric"
    apodisation_w: float | None = None
    kernel_size: int | None = None
    _pupils: dict = field(default_factory=dict, repr=False)

    @property
    def aberrations(self) -> bool:
        return self.combine == "wave_optics"

    def pupil(self, image_height_mm: float) -> FieldPupil:
        key = round(image_height_mm, 6)
        if key not in self._pupils:
            self._pupils[key] = trace_field_pupil(
                self.lens,
                self.z_obj,
                self.z_film,
                key,
                n_pupil=self.n_pupil,
                pupil_amplitude=self.pupil_amplitude,
                apodisation_w=self.apodisation_w,
            )
        return self._pupils[key]

    def tile_centres_px(self) -> tuple[np.ndarray, np.ndarray]:
        w, h = self.film_res
        nx, ny = self.tile_grid
        return (np.arange(nx) + 0.5) * w / nx, (np.arange(ny) + 0.5) * h / ny

    def tile_kernel(self, cx_px: float, cy_px: float, lam_nm: float, kernel_size: int) -> np.ndarray:
        """Raster kernel [d_row, d_col] for the tile centred at full-film pixel (cx, cy)."""
        w, h = self.film_res
        dc, dr = cx_px - 0.5 * w, cy_px - 0.5 * h
        # pbrt RealisticCamera raster -> film: x_film = -dc p, y_film = +dr p.
        fx, fy = -dc * self.pitch_um * 1e-3, dr * self.pitch_um * 1e-3
        r = math.hypot(fx, fy)
        fp = self.pupil(r)
        delta = (math.atan2(fy, fx) + 0.5 * math.pi) if r > 0 else 0.0
        m = _rotation(-delta) @ np.diag([-1.0, 1.0])
        return pixel_kernel(fp, lam_nm, self.pitch_um, kernel_size, aberrations=self.aberrations, m=m)

    def kernel_size_for(self, lam_nm: float) -> int:
        if self.kernel_size:
            return int(self.kernel_size) | 1
        w, h = self.film_res
        corner = 1e-3 * self.pitch_um * 0.5 * math.hypot(w, h) * (1 - 1 / max(self.tile_grid))
        sizes = [
            auto_kernel_size(self.pupil(rr), lam_nm, self.pitch_um, self.aberrations)
            for rr in np.linspace(0.0, corner, 3)
        ]
        return max(sizes)


def _hat_weights(n_pix: int, offset: int, centres: np.ndarray) -> np.ndarray:
    """(n_tiles, n_pix) linear-interpolation weights of pixel centres between tile centres."""
    x = offset + np.arange(n_pix) + 0.5
    eye = np.eye(centres.size)
    return np.stack([np.interp(x, centres, eye[k]) for k in range(centres.size)])


def apply_spatially_varying(img: np.ndarray, kernels: dict, wy: np.ndarray, wx: np.ndarray, pad: int) -> np.ndarray:
    """out = sum_k h_k * (w_k img), w_k = wy[j] x wx[i]; reflect padding of ``pad`` pixels."""
    from scipy.signal import fftconvolve

    h_img, w_img = img.shape
    fp = np.pad(np.asarray(img, dtype=np.float64), pad, mode="reflect")
    wyp = np.pad(wy, ((0, 0), (pad, pad)), mode="edge")
    wxp = np.pad(wx, ((0, 0), (pad, pad)), mode="edge")
    out = np.zeros_like(fp)
    for (j, i), ker in kernels.items():
        ys, xs = np.flatnonzero(wyp[j] > 0), np.flatnonzero(wxp[i] > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
        src = fp[y0:y1, x0:x1] * np.outer(wyp[j, y0:y1], wxp[i, x0:x1])
        conv = fftconvolve(src, ker, mode="full")
        hk = ker.shape[0] // 2
        oy0, ox0 = y0 - hk, x0 - hk
        cy0, cx0 = max(0, -oy0), max(0, -ox0)
        cy1 = conv.shape[0] - max(0, oy0 + conv.shape[0] - out.shape[0])
        cx1 = conv.shape[1] - max(0, ox0 + conv.shape[1] - out.shape[1])
        out[oy0 + cy0 : oy0 + cy1, ox0 + cx0 : ox0 + cx1] += conv[cy0:cy1, cx0:cx1]
    return out[pad : pad + h_img, pad : pad + w_img].astype(np.float32)


def apply_lens_psf(
    channels: dict[str, np.ndarray],
    model: LensPsfModel,
    *,
    crop_origin: tuple[int, int] = (0, 0),
    wavelength_of=None,
    log=None,
) -> dict[str, np.ndarray]:
    """Convolve every channel (S0.* buckets at their centre wavelength) with the lens PSF."""
    from exr_multispectral import parse_s0_wavelength_nm

    first = next(iter(channels.values()))
    h_img, w_img = first.shape
    cx, cy = model.tile_centres_px()
    wx = _hat_weights(w_img, crop_origin[0], cx)
    wy = _hat_weights(h_img, crop_origin[1], cy)
    tiles = [(j, i) for j in range(cy.size) if wy[j].any() for i in range(cx.size) if wx[i].any()]
    out = {}
    for name, arr in channels.items():
        lam = (wavelength_of or (lambda n: parse_s0_wavelength_nm(n) or RGB_CENTER_NM.get(n.upper())))(name)
        if lam is None:
            out[name] = arr
            continue
        k = model.kernel_size_for(lam)
        kernels = {(j, i): model.tile_kernel(cx[i], cy[j], lam, k) for (j, i) in tiles}
        out[name] = apply_spatially_varying(arr, kernels, wy, wx, pad=k // 2 + 1)
        if log:
            log(f"  {name}: lambda={lam:.1f} nm kernel {k}x{k}, {len(tiles)} tiles")
    return out


def model_from_config(
    cfg: dict, lens_cfg: dict, sensor_cfg: dict, *, repo: Path, film_res: tuple[int, int]
) -> LensPsfModel:
    """Build a :class:`LensPsfModel` from ``lens.post_psf.lens_diffraction`` (+ lens / sensor)."""
    lensfile = cfg.get("lensfile", lens_cfg.get("realistic_lensfile"))
    if not lensfile:
        raise KeyError("lens_diffraction needs lensfile (or lens.realistic_lensfile)")
    lens_path = Path(lensfile) if Path(lensfile).is_absolute() else repo / lensfile
    ap = cfg.get("aperture_diameter_mm", lens_cfg.get("realistic_aperture_diameter_mm"))
    lens = LensSystem.from_file(
        lens_path,
        None if ap is None else float(ap),
        aperture_blades=int(cfg.get("aperture_blades", 0)),
        blade_rotation_deg=float(cfg.get("blade_rotation_deg", 0.0)),
    )
    combine = str(cfg.get("combine", "diffraction_only")).lower()
    if combine not in ("diffraction_only", "wave_optics"):
        raise ValueError('lens_diffraction.combine must be "diffraction_only" or "wave_optics"')
    w, h = film_res
    if cfg.get("pixel_pitch_um") is not None:
        pitch = float(cfg["pixel_pitch_um"])
        diag = pitch * 1e-3 * math.hypot(w, h)
    else:
        diag = float(cfg.get("film_diagonal_mm", PBRT_DEFAULT_FILM_DIAGONAL_MM))
        pitch = 1e3 * diag / math.hypot(w, h)
    focus_m = cfg.get("focus_distance_m", lens_cfg.get("realistic_focus_distance"))
    if focus_m is None:
        raise KeyError("lens_diffraction needs focus_distance_m (or lens.realistic_focus_distance)")
    obj_m = float(cfg.get("object_distance_m", focus_m))
    t_film = pbrt_film_distance_mm(lens, 1e3 * float(focus_m), diag)
    z_film = float(lens.z_vertex[-1]) + t_film
    tg = cfg.get("tile_grid", [9, 7])
    return LensPsfModel(
        lens=lens,
        z_film=z_film,
        z_obj=z_film - 1e3 * obj_m,
        pitch_um=pitch,
        film_res=(int(w), int(h)),
        combine=combine,
        tile_grid=(int(tg[0]), int(tg[1])),
        n_pupil=int(cfg.get("pupil_samples", 97)) | 1,
        pupil_amplitude=str(cfg.get("pupil_amplitude", "radiometric")),
        apodisation_w=None if cfg.get("apodisation_w") is None else float(cfg["apodisation_w"]),
        kernel_size=cfg.get("kernel_size_px"),
    )


def apply_post_psf_config(
    channels: dict[str, np.ndarray], psf_cfg: dict, camera_model: dict, *, repo: Path, exr_path: Path
) -> dict[str, np.ndarray]:
    """Hook for ``apply_spectral_psf`` (``post_psf.mode: lens_diffraction``).

    Pixel pitch: ``lens_diffraction.pixel_pitch_um`` > ``lens_diffraction.film_diagonal_mm`` (must
    equal the pbrt Film "diagonal"); otherwise pbrt's 35 mm default film diagonal for ``realistic``
    renders and ``sensor.pixel_pitch_um`` for pinhole/perspective renders.
    """
    lens_cfg = camera_model.get("lens", {}) or {}
    sensor_cfg = camera_model.get("sensor", {}) or {}
    cfg = dict(psf_cfg.get("lens_diffraction", {}) or {})
    if "pixel_pitch_um" not in cfg and "film_diagonal_mm" not in cfg:
        if str(lens_cfg.get("camera", "")).lower() == "realistic":
            cfg["film_diagonal_mm"] = PBRT_DEFAULT_FILM_DIAGONAL_MM
        elif sensor_cfg.get("pixel_pitch_um") is not None:
            cfg["pixel_pitch_um"] = float(sensor_cfg["pixel_pitch_um"])
    film_res, crop = exr_windows(exr_path)
    model = model_from_config(cfg, lens_cfg, sensor_cfg, repo=repo, film_res=film_res)
    print(
        f"lens_diffraction: {model.combine}, pitch {model.pitch_um:.3f} um, "
        f"tiles {model.tile_grid}, film {film_res}, crop origin {crop}"
    )
    return apply_lens_psf(channels, model, crop_origin=crop, log=print)


def exr_windows(path: Path) -> tuple[tuple[int, int], tuple[int, int]]:
    """(full film resolution (W, H), crop origin (x0, y0)) from EXR display/data windows."""
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        hdr = f.header()
        (x0, y0), (x1, y1) = (tuple(int(v) for v in w) for w in hdr["dataWindow"])
        disp = hdr.get("displayWindow", ((x0, y0), (x1, y1)))
        (dx0, dy0), (dx1, dy1) = (tuple(int(v) for v in w) for w in disp)
    return (int(dx1 - dx0 + 1), int(dy1 - dy0 + 1)), (int(x0 - dx0), int(y0 - dy0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply traced-lens diffraction PSFs to a pbrt spectral EXR.")
    ap.add_argument("--exr-in", type=Path, required=True)
    ap.add_argument("--exr-out", type=Path, required=True)
    ap.add_argument("--lensfile", type=Path, required=True)
    ap.add_argument("--aperture-diameter-mm", type=float, default=None)
    ap.add_argument("--focus-distance-m", type=float, required=True)
    ap.add_argument("--object-distance-m", type=float, default=None)
    ap.add_argument("--combine", choices=("diffraction_only", "wave_optics"), default="diffraction_only")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--film-diagonal-mm", type=float, default=None)
    g.add_argument("--pixel-pitch-um", type=float, default=None)
    ap.add_argument("--tile-grid", type=int, nargs=2, default=(9, 7))
    ap.add_argument("--pupil-samples", type=int, default=97)
    ap.add_argument("--kernel-size-px", type=int, default=None)
    args = ap.parse_args()
    from exr_multispectral import read_separate_exr_channels, write_separate_channels_exr

    film_res, crop = exr_windows(args.exr_in)
    cfg = {
        "lensfile": str(args.lensfile.resolve()),
        "aperture_diameter_mm": args.aperture_diameter_mm,
        "focus_distance_m": args.focus_distance_m,
        "object_distance_m": args.object_distance_m or args.focus_distance_m,
        "combine": args.combine,
        "tile_grid": args.tile_grid,
        "pupil_samples": args.pupil_samples,
        "kernel_size_px": args.kernel_size_px,
    }
    if args.pixel_pitch_um is not None:
        cfg["pixel_pitch_um"] = args.pixel_pitch_um
    if args.film_diagonal_mm is not None:
        cfg["film_diagonal_mm"] = args.film_diagonal_mm
    model = model_from_config(cfg, {}, {}, repo=Path.cwd(), film_res=film_res)
    print(f"lens_diffraction: {args.combine}, pitch {model.pitch_um:.3f} um, film {film_res}, crop origin {crop}")
    chans = read_separate_exr_channels(args.exr_in)
    out = apply_lens_psf(chans, model, crop_origin=crop, log=print)
    write_separate_channels_exr(args.exr_out, out)
    print(f"wrote {args.exr_out}")


if __name__ == "__main__":
    sys.exit(main())
