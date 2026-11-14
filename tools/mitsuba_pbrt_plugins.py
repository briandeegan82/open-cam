"""Mitsuba 3 plugins for pbrt-v4 features that Mitsuba has no core plugin for (tools/pbrt_to_mitsuba.py).

* ``pbrt_gonio_set``: all goniometric point lights in one emitter (pbrt GoniometricLight: equal-area image,
                      nearest lookup, y and z swapped in light space), chosen by power or by intensity / d^2
* ``pbrt_area_angular``: diffuse area light with an equal-area angular map (open-cam patch 0016)
* ``pbrt_retro``    : retroreflective BSDF (open-cam pbrt patches 0001/0033: Lambert plus an observation-angle
                      lobe)
* ``pbrt_difftrans``: diffuse reflection plus diffuse transmission (pbrt DiffuseTransmissionBxDF)
* ``pbrt_scale``, ``pbrt_mix``, ``pbrt_planar``: pbrt's ``scale`` and ``mix`` textures and ``planar`` mapping

Written for the JIT variants (``cuda_ad_spectral``): every method is plain Dr.Jit arithmetic, no Python branching on
per-lane values. Call :func:`register` after ``mi.set_variant``.
"""

from __future__ import annotations

import math

import drjit as dr
import mitsuba as mi
import numpy as np

INV_PI = 1.0 / math.pi


def equal_area_sphere_to_square(d):
    """pbrt's EqualAreaSphereToSquare (Clarberg) for unit vectors ``d``; returns ``(u, v)`` in [0, 1]."""
    x, y, z = dr.abs(d.x), dr.abs(d.y), dr.abs(d.z)
    r = dr.safe_sqrt(1 - z)
    a = dr.maximum(x, y)
    b = dr.select(a == 0, 0, dr.minimum(x, y) / dr.maximum(a, 1e-30))
    phi = dr.atan(b) * (2 / math.pi)
    phi = dr.select(x < y, 1 - phi, phi)
    v = phi * r
    u = r - v
    south = d.z < 0
    u, v = dr.select(south, 1 - v, u), dr.select(south, 1 - u, v)
    u = dr.copysign(u, d.x)
    v = dr.copysign(v, d.y)
    return 0.5 * (u + 1), 0.5 * (v + 1)


def _matrix(t) -> np.ndarray:
    """4x4 float64 matrix of a (scalar, double) Mitsuba transform property."""
    return np.array(t.matrix, dtype=np.float64).reshape(4, 4)


def _spectrum_mean(s):
    return (s[0] + s[1] + s[2] + s[3]) * 0.25 if dr.size_v(s) == 4 else dr.mean(s, axis=0)


def _spectrum_max(s):
    m = s[0]
    for i in range(1, dr.size_v(s)):
        m = dr.maximum(m, s[i])
    return m


class GoniometricLights(mi.Emitter):
    """All of a scene's goniometric point lights as one emitter (pbrt GoniometricLight per lamp).

    One Python emitter instead of one per lamp: Mitsuba re-traces Python plugins on every ``mi.render`` call,
    per instance. Lamp intensity: ``I(lambda) * image(EqualAreaSphereToSquare(w_light))`` (nearest lookup, light
    space = pbrt CTM times the y/z swap). ``filename`` names a JSON list of lamps: ``to_world`` (4x4), ``image``,
    ``spd`` ([[lambda, value], ...], already divided by its photometric integral), ``weight_scale`` (the
    luminous factor of the spd, for selection) and ``phi`` (luminous power).

    ``select``: ``power`` picks a lamp in proportion to its power (Mitsuba's own emitter choice);
    ``exact`` in proportion to its luminous intensity toward the shading point over the squared distance
    (like pbrt patch 0024's exact goniometric sampler; costs one image lookup per lamp).
    """

    def __init__(self, props):
        mi.Emitter.__init__(self, props)
        import json  # noqa: PLC0415

        from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415

        lamps = json.loads(open(props["filename"]).read())
        self.select = props.get("select", "power")
        if self.select not in ("power", "exact"):
            raise ValueError(f"select must be power or exact, got {self.select}")
        images, specs, offsets = {}, {}, []
        buf = []
        size = 0
        for lp in lamps:
            if lp["image"] not in images:
                ch = read_separate_exr_channels(lp["image"])
                img = np.ascontiguousarray(ch["Y"] if "Y" in ch else next(iter(ch.values())), dtype=np.float32)
                images[lp["image"]] = (size, img.shape)
                buf.append(img.ravel())
                size += img.size
            key = json.dumps(lp["spd"])
            if key not in specs:
                a = np.asarray(lp["spd"], float)
                specs[key] = (len(specs), mi.load_dict({"type": "irregular",
                                                       "wavelengths": ",".join(f"{x:.6g}" for x in a[:, 0]),
                                                       "values": ",".join(f"{x:.9g}" for x in a[:, 1])}))
        shapes = {sh for _, sh in images.values()}
        if len(shapes) != 1 or next(iter(shapes))[0] != next(iter(shapes))[1]:
            raise ValueError(f"goniometric images must share one square size, got {shapes}")
        self.res = next(iter(shapes))[0]
        self.image = mi.Float(np.concatenate(buf))
        self.spectra = [t for _, t in sorted(specs.values(), key=lambda x: x[0])]
        self.n = len(lamps)
        to_local = [np.linalg.inv(np.asarray(lp["to_world"], float)) for lp in lamps]
        pos = np.array([np.asarray(lp["to_world"], float)[:3, 3] for lp in lamps])
        # per-lamp scalars, baked into the trace for the exact choice
        # plain floats: a numpy scalar times a symbolic Dr.Jit variable would try to evaluate it
        self.l_pos = [[float(x) for x in p] for p in pos]
        self.l_rot = [[[float(x) for x in row] for row in m[:3, :3]] for m in to_local]
        self.l_off = [int(images[lp["image"]][0]) for lp in lamps]
        self.l_spec = [specs[json.dumps(lp["spd"])][0] for lp in lamps]
        self.l_ws = [float(lp["weight_scale"]) for lp in lamps]
        # gather tables for the chosen lamp
        self.g_pos = mi.Point3f(pos[:, 0], pos[:, 1], pos[:, 2])
        rot = np.array(self.l_rot)
        self.g_rot = [mi.Float(rot[:, r, c].astype(np.float32)) for r in range(3) for c in range(3)]
        self.g_off = mi.UInt32(np.array(self.l_off, np.uint32))
        self.g_spec = mi.UInt32(np.array(self.l_spec, np.uint32))
        self.g_ws = mi.Float(np.array(self.l_ws, np.float32))
        phi = np.array([lp["phi"] for lp in lamps], float)
        self.power_distr = mi.DiscreteDistribution(phi / phi.sum())
        lo, hi = pos.min(0), pos.max(0)
        self.box = mi.ScalarBoundingBox3f(mi.ScalarPoint3f(*lo), mi.ScalarPoint3f(*hi))
        self.m_flags = mi.EmitterFlags.DeltaPosition

    def _lookup(self, off, wl):
        u, v = equal_area_sphere_to_square(wl)
        x = dr.clip(mi.UInt32(dr.maximum(u * self.res, 0)), 0, self.res - 1)
        y = dr.clip(mi.UInt32(dr.maximum(v * self.res, 0)), 0, self.res - 1)
        return dr.gather(mi.Float, self.image, off + y * self.res + x)

    def _spectrum(self, idx_spec, wavelengths, active):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = wavelengths
        out = self.spectra[0].eval(si, active)
        for j in range(1, len(self.spectra)):
            out = dr.select(idx_spec == j, self.spectra[j].eval(si, active), out)
        return out

    def _exact_weights(self, p):
        """Selection weight of every lamp toward ``p`` (Python loop: unrolled into the kernel)."""
        ws = []
        for i in range(self.n):
            d = mi.Vector3f(p.x - self.l_pos[i][0], p.y - self.l_pos[i][1], p.z - self.l_pos[i][2])
            d2 = dr.maximum(dr.squared_norm(d), 1e-8)
            R = self.l_rot[i]
            wl = mi.Vector3f(R[0][0] * d.x + R[0][1] * d.y + R[0][2] * d.z,
                             R[1][0] * d.x + R[1][1] * d.y + R[1][2] * d.z,
                             R[2][0] * d.x + R[2][1] * d.y + R[2][2] * d.z) * dr.rsqrt(d2)
            ws.append(self.l_ws[i] * self._lookup(self.l_off[i], dr.normalize(wl)) / d2)
        return ws

    def sample_direction(self, it, sample, active=True):
        if self.select == "power":
            idx, _ = self.power_distr.sample_reuse(sample.x, active)
            pmf = self.power_distr.eval_pmf_normalized(idx, active)
        else:
            ws = self._exact_weights(it.p)
            # no "+=": it would modify ws[0] in place
            total = ws[0]
            for w in ws[1:]:
                total = total + w
            u = sample.x * total
            idx = mi.UInt32(0)
            pmf_w = mi.Float(0)
            acc = mi.Float(0)
            for i, w in enumerate(ws):
                take = (u >= acc) & (w > 0)
                idx = dr.select(take, mi.UInt32(i), idx)
                pmf_w = dr.select(take, w, pmf_w)
                acc = acc + w
            pmf = dr.select(total > 0, pmf_w / dr.maximum(total, 1e-30), 0)
        pos = dr.gather(mi.Point3f, self.g_pos, idx, active)
        R = [dr.gather(mi.Float, g, idx, active) for g in self.g_rot]
        d = pos - it.p
        dist2 = dr.squared_norm(d)
        dist = dr.sqrt(dist2)
        w = -d / dist
        wl = dr.normalize(mi.Vector3f(R[0] * w.x + R[1] * w.y + R[2] * w.z,
                                      R[3] * w.x + R[4] * w.y + R[5] * w.z,
                                      R[6] * w.x + R[7] * w.y + R[8] * w.z))
        I = self._lookup(dr.gather(mi.UInt32, self.g_off, idx, active), wl)
        spec = self._spectrum(dr.gather(mi.UInt32, self.g_spec, idx, active), it.wavelengths, active)
        ok = active & (pmf > 0)
        ds = dr.zeros(mi.DirectionSample3f)
        ds.p = pos
        ds.n = mi.Normal3f(0, 0, 0)
        ds.uv = mi.Point2f(0, 0)
        ds.time = it.time
        ds.pdf = pmf
        ds.delta = True
        ds.emitter = mi.EmitterPtr(self)
        ds.d = d / dist
        ds.dist = dist
        val = spec * (I / (dist2 * dr.maximum(pmf, 1e-30)))
        return ds, dr.select(ok, val, 0)

    def pdf_direction(self, it, ds, active=True):
        return mi.Float(0)

    def eval_direction(self, it, ds, active=True):
        return mi.Spectrum(0)

    def eval(self, si, active=True):
        return mi.Spectrum(0)

    def sample_wavelengths(self, si, sample, active=True):
        return self.spectra[0].sample_spectrum(si, mi.sample_shifted(sample), active)

    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        raise NotImplementedError("pbrt_gonio_set supports only next-event estimation (path integrator)")

    def bbox(self):
        return self.box

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return f"GoniometricLights[{self.n} lamps, select={self.select}]"


class AngularAreaLight(mi.Emitter):
    """open-cam pbrt patch 0016: diffuse area light whose radiance is scaled by an equal-area ``angularfilename``
    map in light space (bilinear lookup). One-sided (shading normal side). Nest it in a shape.

    Properties: ``radiance`` (spectrum texture), ``filename`` (map, first channel), ``to_world`` (light space),
    ``lamp_id`` (any unique string: keeps Mitsuba from sharing one instance between identical lamps).
    """

    def __init__(self, props):
        mi.Emitter.__init__(self, props)
        from exr_multispectral import read_separate_exr_channels  # noqa: PLC0415

        ch = read_separate_exr_channels(props["filename"])
        img = np.ascontiguousarray(ch["Y"] if "Y" in ch else next(iter(ch.values())), dtype=np.float32)
        self.h, self.w = img.shape
        self.image = mi.Float(img.ravel())
        self.radiance = props["radiance"]
        self.lamp_id = props.get("lamp_id", "")
        m = _matrix(props["to_world"])
        self.to_local = mi.Transform4f(np.linalg.inv(m).astype(np.float32))
        self.m_flags = mi.EmitterFlags.Surface

    def _angular(self, w_world):
        u, v = equal_area_sphere_to_square(dr.normalize(self.to_local @ w_world))
        x = u * self.w - 0.5
        y = v * self.h - 0.5
        x0, y0 = dr.floor(x), dr.floor(y)
        fx, fy = x - x0, y - y0
        xi0 = dr.clip(mi.Int32(x0), 0, self.w - 1)
        xi1 = dr.clip(mi.Int32(x0) + 1, 0, self.w - 1)
        yi0 = dr.clip(mi.Int32(y0), 0, self.h - 1)
        yi1 = dr.clip(mi.Int32(y0) + 1, 0, self.h - 1)
        g = lambda xi, yi: dr.gather(mi.Float, self.image, yi * self.w + xi)  # noqa: E731
        return (g(xi0, yi0) * (1 - fx) + g(xi1, yi0) * fx) * (1 - fy) + (g(xi0, yi1) * (1 - fx) + g(xi1, yi1) * fx) * fy

    def _L(self, wavelengths, w_out, active):
        si = dr.zeros(mi.SurfaceInteraction3f)
        si.wavelengths = wavelengths
        return self.radiance.eval(si, active) * self._angular(w_out)

    def eval(self, si, active=True):
        ok = active & (mi.Frame3f.cos_theta(si.wi) > 0)
        return dr.select(ok, self._L(si.wavelengths, si.to_world(si.wi), ok), 0)

    def sample_direction(self, it, sample, active=True):
        ds = self.get_shape().sample_direction(it, sample, active)
        ok = active & (dr.dot(ds.d, ds.n) < 0) & (ds.pdf > 0)
        ds.pdf = dr.select(ok, ds.pdf, 0)
        ds.emitter = mi.EmitterPtr(self)
        val = self._L(it.wavelengths, -ds.d, ok) / dr.maximum(ds.pdf, 1e-30)
        return ds, dr.select(ok, val, 0)

    def pdf_direction(self, it, ds, active=True):
        ok = active & (dr.dot(ds.d, ds.n) < 0)
        return dr.select(ok, self.get_shape().pdf_direction(it, ds, ok), 0)

    def eval_direction(self, it, ds, active=True):
        ok = active & (dr.dot(ds.d, ds.n) < 0)
        return dr.select(ok, self._L(it.wavelengths, -ds.d, ok), 0)

    def sample_wavelengths(self, si, sample, active=True):
        return self.radiance.sample_spectrum(si, mi.sample_shifted(sample), active)

    def sample_ray(self, time, sample1, sample2, sample3, active=True):
        raise NotImplementedError("pbrt_area_angular supports only next-event estimation (path integrator)")

    def bbox(self):
        return self.get_shape().bbox()

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return f"AngularAreaLight[{self.w}x{self.h}]"


class RetroreflectiveBSDF(mi.BSDF):
    """open-cam pbrt ``retroreflective``: ``R/pi`` plus a lobe in the observation angle (front side only).

    Properties: ``reflectance`` (R, clamped to [0, 1]), ``ra`` (lobe spectrum, clamped >= 0), ``alpha0`` and ``width``
    [rad], ``q`` (entrance exponent). Wrap in ``twosided``.
    """

    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.R = props["reflectance"]
        self.ra = props["ra"]
        self.alpha0 = float(props["alpha0"])
        self.width = float(props["width"])
        self.q = float(props["q"])
        flags = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        self.m_components = [mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide,
                             mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide]
        self.m_flags = flags

    def _tex(self, si, active):
        R = dr.clip(self.R.eval(si, active), 0, 1)
        ra = dr.maximum(self.ra.eval(si, active), 0)
        return R, ra

    def _energy_scale(self, R, ra, c):
        peak = _spectrum_max(ra) * math.exp(self.alpha0 / self.width)
        albedo = peak * 2 * math.pi * self.width**2 * dr.power(dr.maximum(c, 1e-4), self.q - 1)
        budget = dr.maximum(0, 1 - _spectrum_max(R))
        return dr.select(albedo > budget, budget / dr.maximum(albedo, 1e-30), 1)

    @staticmethod
    def _obs_angle(wi, wo):
        return 2 * dr.safe_asin(dr.norm(wi - wo) / 2)

    def _f(self, R, ra, wi, wo):
        ci, co = mi.Frame3f.cos_theta(wo), mi.Frame3f.cos_theta(wi)
        ok = (ci > 0) & (co > 0)
        alpha = self._obs_angle(wi, wo)
        c = dr.maximum(ci * co, 1e-12)
        g = dr.exp(-(alpha - self.alpha0) / self.width) * dr.power(c, self.q / 2 - 1)
        lobe = ra * (g * self._energy_scale(R, ra, dr.sqrt(c)))
        return dr.select(ok, R * INV_PI + lobe, 0)

    def _lobe_prob(self, R, ra, co):
        peak = _spectrum_mean(ra) * math.exp(self.alpha0 / self.width)
        albedo = peak * 2 * math.pi * self.width**2 * dr.power(dr.maximum(co, 1e-12), self.q - 1) \
            * self._energy_scale(R, ra, co)
        d = _spectrum_mean(R)
        p = dr.maximum(0.25, albedo / dr.maximum(albedo + d, 1e-30))
        p = dr.select(d <= 0, 1, p)
        return dr.select((albedo + d <= 0) | (co <= 0), 0, p)

    def _lobe_pdf(self, wi, wo):
        alpha = self._obs_angle(wi, wo)
        s = dr.sin(alpha)
        return dr.select(s > 0, alpha * dr.exp(-alpha / self.width) / (self.width**2 * 2 * math.pi * dr.maximum(s, 1e-30)), 0)

    def _pdf(self, R, ra, wi, wo):
        pr = self._lobe_prob(R, ra, mi.Frame3f.cos_theta(wi))
        cw = mi.Frame3f.cos_theta(wo)
        p = (1 - pr) * dr.maximum(cw, 0) * INV_PI + pr * self._lobe_pdf(wi, wo)
        return dr.select((cw > 0) & (mi.Frame3f.cos_theta(wi) > 0), p, 0)

    def sample(self, ctx, si, sample1, sample2, active=True):
        R, ra = self._tex(si, active)
        wi = si.wi
        pr = self._lobe_prob(R, ra, mi.Frame3f.cos_theta(wi))
        use_lobe = sample1 < pr
        u0 = dr.maximum(sample1 / dr.maximum(pr, 1e-30), 1e-7)
        alpha = -self.width * dr.log(dr.maximum(u0 * sample2.x, 1e-12))
        phi = 2 * math.pi * sample2.y
        sa, ca = dr.sin(alpha), dr.cos(alpha)
        frame = mi.Frame3f(wi)
        w_lobe = frame.to_world(mi.Vector3f(sa * dr.cos(phi), sa * dr.sin(phi), ca))
        w_diff = mi.warp.square_to_cosine_hemisphere(sample2)
        wo = dr.select(use_lobe, w_lobe, w_diff)
        pdf = self._pdf(R, ra, wi, wo)
        ok = active & (pdf > 0) & (mi.Frame3f.cos_theta(wo) > 0) & (mi.Frame3f.cos_theta(wi) > 0) \
            & ~(use_lobe & (alpha >= math.pi))
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo
        bs.pdf = pdf
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(dr.select(use_lobe, mi.UInt32(+mi.BSDFFlags.GlossyReflection),
                                              mi.UInt32(+mi.BSDFFlags.DiffuseReflection)))
        bs.sampled_component = dr.select(use_lobe, mi.UInt32(1), mi.UInt32(0))
        val = self._f(R, ra, wi, wo) * mi.Frame3f.cos_theta(wo) / dr.maximum(pdf, 1e-30)
        return bs, dr.select(ok, val, 0)

    def eval(self, ctx, si, wo, active=True):
        R, ra = self._tex(si, active)
        return dr.select(active, self._f(R, ra, si.wi, wo) * dr.maximum(mi.Frame3f.cos_theta(wo), 0), 0)

    def pdf(self, ctx, si, wo, active=True):
        R, ra = self._tex(si, active)
        return dr.select(active, self._pdf(R, ra, si.wi, wo), 0)

    def eval_pdf(self, ctx, si, wo, active=True):
        R, ra = self._tex(si, active)
        f = self._f(R, ra, si.wi, wo) * dr.maximum(mi.Frame3f.cos_theta(wo), 0)
        return dr.select(active, f, 0), dr.select(active, self._pdf(R, ra, si.wi, wo), 0)

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return f"RetroreflectiveBSDF[alpha0={self.alpha0}, width={self.width}, q={self.q}]"


class DiffuseTransmissionBSDF(mi.BSDF):
    """``reflectance/pi`` on the incident side, ``transmittance/pi`` on the other (both sides; no ``twosided``)."""

    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.r = props["reflectance"]
        self.t = props["transmittance"]
        refl = mi.BSDFFlags.DiffuseReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        trans = mi.BSDFFlags.DiffuseTransmission | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
        self.m_components = [refl, trans]
        self.m_flags = refl | trans

    def _rt(self, si, active):
        return dr.clip(self.r.eval(si, active), 0, 1), dr.clip(self.t.eval(si, active), 0, 1)

    @staticmethod
    def _pr(r, t):
        a, b = _spectrum_mean(r), _spectrum_mean(t)
        return dr.select(a + b > 0, a / dr.maximum(a + b, 1e-30), 0.5)

    def _f_pdf(self, si, wo, active):
        r, t = self._rt(si, active)
        ci, co = mi.Frame3f.cos_theta(si.wi), mi.Frame3f.cos_theta(wo)
        same = ci * co > 0
        pr = self._pr(r, t)
        f = dr.select(same, r, t) * INV_PI * dr.abs(co)
        pdf = dr.select(same, pr, 1 - pr) * dr.abs(co) * INV_PI
        return dr.select(active & (ci != 0), f, 0), dr.select(active & (ci != 0), pdf, 0)

    def sample(self, ctx, si, sample1, sample2, active=True):
        r, t = self._rt(si, active)
        pr = self._pr(r, t)
        refl = sample1 < pr
        w = mi.warp.square_to_cosine_hemisphere(sample2)
        side = dr.select(mi.Frame3f.cos_theta(si.wi) >= 0, 1.0, -1.0)
        w.z = dr.select(refl, w.z * side, -w.z * side)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = w
        bs.eta = 1.0
        bs.pdf = dr.select(refl, pr, 1 - pr) * dr.abs(w.z) * INV_PI
        bs.sampled_type = mi.UInt32(dr.select(refl, mi.UInt32(+mi.BSDFFlags.DiffuseReflection),
                                              mi.UInt32(+mi.BSDFFlags.DiffuseTransmission)))
        bs.sampled_component = dr.select(refl, mi.UInt32(0), mi.UInt32(1))
        val = dr.select(refl, r / dr.maximum(pr, 1e-30), t / dr.maximum(1 - pr, 1e-30))
        return bs, dr.select(active & (bs.pdf > 0), val, 0)

    def eval(self, ctx, si, wo, active=True):
        return self._f_pdf(si, wo, active)[0]

    def pdf(self, ctx, si, wo, active=True):
        return self._f_pdf(si, wo, active)[1]

    def eval_pdf(self, ctx, si, wo, active=True):
        return self._f_pdf(si, wo, active)

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return "DiffuseTransmissionBSDF[]"


class _Tex(mi.Texture):
    def is_spatially_varying(self):
        return True

    def traverse(self, callback):
        pass

    def parameters_changed(self, keys):
        pass


class ScaleTexture(_Tex):
    """pbrt ``scale``: ``tex * scale`` (either may be a float or spectrum texture)."""

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.a, self.b = props["tex"], props["scale"]

    def eval(self, si, active=True):
        return self.a.eval(si, active) * self.b.eval_1(si, active)

    def eval_1(self, si, active=True):
        return self.a.eval_1(si, active) * self.b.eval_1(si, active)

    def mean(self):
        return self.a.mean() * self.b.mean()

    def to_string(self):
        return "ScaleTexture[]"


class MixTexture(_Tex):
    """pbrt ``mix``: ``(1 - amount) * tex1 + amount * tex2``."""

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.t1, self.t2, self.amount = props["tex1"], props["tex2"], props["amount"]

    def eval(self, si, active=True):
        a = self.amount.eval_1(si, active)
        return self.t1.eval(si, active) * (1 - a) + self.t2.eval(si, active) * a

    def eval_1(self, si, active=True):
        a = self.amount.eval_1(si, active)
        return self.t1.eval_1(si, active) * (1 - a) + self.t2.eval_1(si, active) * a

    def mean(self):
        a = self.amount.mean()
        return self.t1.mean() * (1 - a) + self.t2.mean() * a

    def to_string(self):
        return "MixTexture[]"


class PlanarTexture(_Tex):
    """pbrt ``planar`` mapping around a child bitmap: ``u = du + p_tex . v1``, ``v = 1 - (dv + p_tex . v2)``.

    The ``1 - v`` matches pbrt's image lookup flip on Mitsuba's top-row-first bitmaps.
    """

    def __init__(self, props):
        mi.Texture.__init__(self, props)
        self.tex = props["tex"]
        self.v1 = mi.Vector3f(props["v1"])
        self.v2 = mi.Vector3f(props["v2"])
        self.du, self.dv = float(props.get("udelta", 0.0)), float(props.get("vdelta", 0.0))
        self.to_tex = mi.Transform4f(np.linalg.inv(_matrix(props["to_world"])).astype(np.float32))

    def _si(self, si):
        p = self.to_tex @ si.p
        s2 = mi.SurfaceInteraction3f(si)
        s2.uv = mi.Point2f(self.du + dr.dot(mi.Vector3f(p), self.v1), 1 - (self.dv + dr.dot(mi.Vector3f(p), self.v2)))
        return s2

    def eval(self, si, active=True):
        return self.tex.eval(self._si(si), active)

    def eval_1(self, si, active=True):
        return self.tex.eval_1(self._si(si), active)

    def mean(self):
        return self.tex.mean()

    def to_string(self):
        return "PlanarTexture[]"


_REGISTERED = False


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    mi.register_emitter("pbrt_gonio_set", lambda p: GoniometricLights(p))
    mi.register_emitter("pbrt_area_angular", lambda p: AngularAreaLight(p))
    mi.register_bsdf("pbrt_retro", lambda p: RetroreflectiveBSDF(p))
    mi.register_bsdf("pbrt_difftrans", lambda p: DiffuseTransmissionBSDF(p))
    mi.register_texture("pbrt_scale", lambda p: ScaleTexture(p))
    mi.register_texture("pbrt_mix", lambda p: MixTexture(p))
    mi.register_texture("pbrt_planar", lambda p: PlanarTexture(p))
    _REGISTERED = True
