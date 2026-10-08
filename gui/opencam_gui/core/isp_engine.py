"""Colour / ISP adapter over ``tools/colour_science.py`` and ``tools/apply_emva_noise.py``.

The demo walks a ColorChecker from spectra to a displayable image one stage at a
time: spectral scene times illuminant times QE, Bayer mosaic, demosaic, white
balance, colour correction matrix, sRGB encode. Every stage can be skipped, and
every stage has a preview, so the cost of each one is visible rather than
asserted.

The scene is synthesised from the real X-Rite reflectance spectra rather than
from RGB values, which is what makes the illuminant sweep meaningful: under F11
the patches genuinely change colour because the light genuinely changes, not
because a constant was swapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opencam_gui.core.repo import import_tool, repo_root

#: 6 across, 4 down, the standard ColorChecker layout.
CHART_COLS, CHART_ROWS = 6, 4
PATCH_PX = 28
DEFAULT_WAVELENGTH_NM = np.arange(380.0, 731.0, 1.0)

DEFAULT_QE_PATHS = {
    "red": "spectra/QE/interpolated/QE_red.csv",
    "green": "spectra/QE/interpolated/QE_green.csv",
    "blue": "spectra/QE/interpolated/QE_blue.csv",
    "ircf": "spectra/QE/interpolated/QE_IRCF.csv",
}

STAGES = ("mosaic", "demosaic", "white_balance", "ccm", "srgb")
STAGE_LABELS = {
    "mosaic": "Bayer mosaic",
    "demosaic": "Demosaic",
    "white_balance": "White balance",
    "ccm": "Colour correction matrix",
    "srgb": "sRGB encode",
}

DEMOSAIC_METHODS = ("bilinear", "malvar")
WB_METHODS = ("gray_world", "white_patch", "none")


def _cs():
    return import_tool("colour_science")


def _noise():
    return import_tool("apply_emva_noise")


def list_illuminants() -> list[str]:
    return _cs().list_illuminant_ids(repo_root())


def qe_paths_from_model(model: dict) -> dict[str, str]:
    """Pull the QE curve paths out of a camera recipe, falling back to defaults."""
    qe = (model.get("sensor", {}) or {}).get("quantum_efficiency", {}) or {}
    paths = {
        "red": qe.get("red_csv") or DEFAULT_QE_PATHS["red"],
        "green": qe.get("green_csv") or DEFAULT_QE_PATHS["green"],
        "blue": qe.get("blue_csv") or DEFAULT_QE_PATHS["blue"],
    }
    if qe.get("ircf_csv"):
        paths["ircf"] = qe["ircf_csv"]
    return paths


def qe_paths_for_recipe(recipe_id: str | None) -> dict[str, str]:
    """Resolve a camera-recipe id to the QE CSV paths ``load_chart`` expects."""
    if not recipe_id:
        return dict(DEFAULT_QE_PATHS)
    from opencam_gui.core.camera import load_camera_model
    from opencam_gui.core.catalog import find_recipe

    return qe_paths_from_model(load_camera_model(find_recipe(recipe_id).path))


# =====================================================================
# Scene: spectra -> camera RGB and reference colour
# =====================================================================
@dataclass(frozen=True)
class Chart:
    """A ColorChecker rendered two ways: as the camera sees it, and as the eye does."""

    wavelength_nm: np.ndarray
    reflectance: np.ndarray  # (24, K)
    illuminant: np.ndarray  # (K,)
    illuminant_id: str
    qe_rgb: np.ndarray  # (3, K)
    camera_rgb: np.ndarray  # (24, 3) raw sensor response
    reference_srgb_linear: np.ndarray  # (24, 3) ground truth, sRGB primaries
    reference_xyz: np.ndarray  # (24, 3)
    names: tuple[str, ...]
    white_point_xyz: np.ndarray
    luther_error: float


def load_chart(
    *,
    illuminant_id: str = "D65",
    qe_paths: dict[str, str] | None = None,
    wavelength_nm: np.ndarray | None = None,
) -> Chart:
    """Integrate the ColorChecker spectra through both the observer and the sensor."""
    cs = _cs()
    repo = repo_root()
    wl = DEFAULT_WAVELENGTH_NM if wavelength_nm is None else np.asarray(wavelength_nm, float)

    chart = cs.load_colorchecker(repo, wl)
    _, spd = cs.load_illuminant(repo, illuminant_id, wl)
    qe = cs.load_qe_rgb(repo, qe_paths or DEFAULT_QE_PATHS, wl)

    # Scale by the green channel's response to a perfect white diffuser. A single
    # scalar, deliberately: normalising each channel against its own white would
    # be a perfect white balance baked into the scene, leaving the white-balance
    # stage nothing to do and no illuminant cast for students to see.
    camera = cs.camera_rgb_from_spectra(wl, chart.reflectance, spd, qe, normalise=False)
    white_response = cs.integrate_spectra(wl, spd[None, :], qe)[0]
    camera = camera / max(float(white_response[1]), 1e-12)

    xyz = cs.xyz_from_spectra(wl, chart.reflectance, spd)

    # The reference is what the scene should look like rendered to sRGB, adapted
    # to D65 so that "correct" means "correct under the display's white point"
    # rather than "tinted by whatever lamp was used".
    white = cs.white_point_xyz(wl, spd)
    adapt = cs.chromatic_adaptation_matrix(white, cs.WHITE_D65)
    reference = cs.xyz_to_srgb_linear(xyz @ adapt.T)

    return Chart(
        wavelength_nm=wl,
        reflectance=chart.reflectance,
        illuminant=spd,
        illuminant_id=illuminant_id,
        qe_rgb=qe,
        camera_rgb=camera,
        reference_srgb_linear=reference,
        reference_xyz=xyz,
        names=chart.names,
        white_point_xyz=white,
        luther_error=float(cs.luther_condition_error(wl, qe)),
    )


def chart_image(values: np.ndarray, *, patch_px: int = PATCH_PX, gap_px: int = 3) -> np.ndarray:
    """Lay 24 per-patch values out as the familiar 6x4 chart image."""
    v = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    h = CHART_ROWS * patch_px + (CHART_ROWS + 1) * gap_px
    w = CHART_COLS * patch_px + (CHART_COLS + 1) * gap_px
    img = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(min(v.shape[0], CHART_ROWS * CHART_COLS)):
        r, c = divmod(i, CHART_COLS)
        y = gap_px + r * (patch_px + gap_px)
        x = gap_px + c * (patch_px + gap_px)
        img[y : y + patch_px, x : x + patch_px] = v[i]
    return img


def patch_mask(*, patch_px: int = PATCH_PX, gap_px: int = 3) -> np.ndarray:
    """Which pixels belong to a patch rather than the gaps between them.

    Fitting a CCM on the gaps would be fitting on black, so the mask matters.
    """
    img = chart_image(np.ones((CHART_ROWS * CHART_COLS, 3)), patch_px=patch_px, gap_px=gap_px)
    return img[:, :, 0] > 0.5


# =====================================================================
# Staged ISP
# =====================================================================
@dataclass
class IspConfig:
    """Which stages run, and how."""

    enabled: set[str] = field(default_factory=lambda: set(STAGES))
    bayer_pattern: str = "RGGB"
    demosaic_method: str = "bilinear"
    wb_method: str = "gray_world"
    exposure_scale: float = 1.0


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline, with the image as it leaves that step."""

    id: str
    label: str
    note: str
    image: np.ndarray  # HxWx3 linear, or HxWx3 display-encoded for the srgb stage
    applied: bool


@dataclass(frozen=True)
class IspResult:
    stages: tuple[Stage, ...]
    ccm: np.ndarray
    wb_gains: np.ndarray
    final_linear: np.ndarray
    final_display: np.ndarray
    reference_display: np.ndarray


def run_isp(chart: Chart, cfg: IspConfig) -> IspResult:
    """Run the staged pipeline and keep a preview after every step."""
    noise = _noise()
    if cfg.demosaic_method not in DEMOSAIC_METHODS:
        raise ValueError(f"unknown demosaic method {cfg.demosaic_method!r}")
    if cfg.wb_method not in WB_METHODS:
        raise ValueError(f"unknown white balance method {cfg.wb_method!r}")

    mask = patch_mask()
    linear = chart_image(chart.camera_rgb) * cfg.exposure_scale
    reference = chart_image(chart.reference_srgb_linear)

    stages: list[Stage] = [
        Stage(
            id="scene",
            label="Sensor linear RGB",
            note=(
                "Spectral reflectance times the illuminant, integrated against the QE curves. "
                "This is the raw response, before the CFA throws two thirds of it away."
            ),
            image=linear.copy(),
            applied=True,
        )
    ]

    # --- mosaic ---------------------------------------------------
    if "mosaic" in cfg.enabled:
        cfa = noise.bayer_sample_rgb(linear.astype(np.float32), cfg.bayer_pattern)
        current = np.repeat(np.asarray(cfa, dtype=np.float64)[:, :, None], 3, axis=2)
        note = (
            f"{cfg.bayer_pattern} colour filter array: each pixel now records one "
            "channel only, so two thirds of the colour information is gone and has "
            "to be guessed back."
        )
        mosaic_applied = True
    else:
        cfa = None
        current = linear.copy()
        note = "Skipped: a three-sensor camera would genuinely measure all three channels here."
        mosaic_applied = False
    stages.append(Stage("mosaic", STAGE_LABELS["mosaic"], note, current.copy(), mosaic_applied))

    # --- demosaic -------------------------------------------------
    if "demosaic" in cfg.enabled and cfa is not None:
        fn = noise.malvar_demosaic if cfg.demosaic_method == "malvar" else noise.bilinear_demosaic
        current = np.asarray(fn(np.asarray(cfa, dtype=np.float32), cfg.bayer_pattern), dtype=np.float64)
        note = (
            f"{cfg.demosaic_method.capitalize()} interpolation reconstructs the two missing "
            "channels at every pixel. Errors here land on edges, as colour fringes."
        )
        applied = True
    else:
        note = (
            "Skipped: without demosaic the mosaic stays as a grey pattern."
            if cfa is not None
            else "Nothing to demosaic -- the mosaic stage was skipped."
        )
        applied = False
    stages.append(Stage("demosaic", STAGE_LABELS["demosaic"], note, current.copy(), applied))

    # --- white balance --------------------------------------------
    gains = np.ones(3, dtype=np.float32)
    if "white_balance" in cfg.enabled and cfg.wb_method != "none":
        # Estimate on the patches only. The gaps between them are black, and
        # letting them into a gray-world average biases the estimate.
        src = current[mask].reshape(1, -1, 3).astype(np.float32)
        gains = noise.white_patch_gains(src) if cfg.wb_method == "white_patch" else noise.gray_world_gains(src)
        current = np.asarray(noise.apply_rgb_gains(current.astype(np.float32), gains), dtype=np.float64)
        note = (
            f"{cfg.wb_method.replace('_', ' ').capitalize()} gains "
            f"R {gains[0]:.2f}, G {gains[1]:.2f}, B {gains[2]:.2f}. "
            "This removes the illuminant's overall cast but cannot fix its shape."
        )
        applied = True
    else:
        note = "Skipped: the image keeps the illuminant's colour cast."
        applied = False
    stages.append(Stage("white_balance", STAGE_LABELS["white_balance"], note, current.copy(), applied))

    # --- ccm ------------------------------------------------------
    ccm = np.eye(3, dtype=np.float32)
    if "ccm" in cfg.enabled:
        # Fit on patch means so every patch counts equally rather than by area,
        # and on exposure-normalised values so the matrix carries colour rather
        # than gain. sanitize_ccm is deliberately not used here: its gain clamp
        # is a guard for the pipeline's preview path and it would cap a
        # legitimately fitted matrix.
        src = _patch_means(current)
        scale = _neutral_exposure_scale(src, chart.reference_srgb_linear)
        ccm = noise.fit_ccm_lstsq(
            (src * scale).astype(np.float32).reshape(1, -1, 3),
            chart.reference_srgb_linear.astype(np.float32).reshape(1, -1, 3),
        )
        current = np.asarray(noise.apply_ccm((current * scale).astype(np.float32), ccm), dtype=np.float64)
        note = (
            "A 3x3 fitted against the colorimetric reference. It corrects what a linear "
            "map can correct; what is left over is the camera failing the Luther "
            "condition, and no 3x3 will remove it."
        )
        applied = True
    else:
        note = "Skipped: raw camera RGB is not a colour space, and it shows."
        applied = False
    stages.append(Stage("ccm", STAGE_LABELS["ccm"], note, current.copy(), applied))

    # --- srgb encode ----------------------------------------------
    final_linear = current.copy()
    if "srgb" in cfg.enabled:
        display = np.asarray(noise.linear_to_srgb(np.clip(current, 0.0, 1.0).astype(np.float32)), dtype=np.float64)
        note = (
            "The sRGB transfer function. Non-linear on purpose: it spends code values "
            "where the eye can tell them apart."
        )
        applied = True
    else:
        display = np.clip(current, 0.0, 1.0)
        note = "Skipped: linear light shown directly looks far too dark in the shadows."
        applied = False
    stages.append(Stage("srgb", STAGE_LABELS["srgb"], note, display.copy(), applied))

    reference_display = np.asarray(
        _noise().linear_to_srgb(np.clip(reference, 0.0, 1.0).astype(np.float32)), dtype=np.float64
    )

    return IspResult(
        stages=tuple(stages),
        ccm=np.asarray(ccm, dtype=np.float64),
        wb_gains=np.asarray(gains, dtype=np.float64),
        final_linear=final_linear,
        final_display=display,
        reference_display=reference_display,
    )


# =====================================================================
# Demosaic comparison
# =====================================================================
@dataclass(frozen=True)
class DemosaicComparison:
    bilinear: np.ndarray
    malvar: np.ndarray
    difference: np.ndarray
    bilinear_rmse: float
    malvar_rmse: float
    edge_bilinear_rmse: float
    edge_malvar_rmse: float


def compare_demosaic(*, pattern: str = "RGGB", size: int = 96) -> DemosaicComparison:
    """Both demosaic methods against a known ground truth, with the error mapped.

    The test image has natural image statistics -- fine luminance detail carried
    at close to constant hue -- because that is the assumption Malvar's
    gradient correction is built on, and the regime where zipper artefacts
    actually bite. Give it saturated, independently varying channels instead and
    Malvar loses to bilinear, which is a statement about the image rather than
    about the algorithm.
    """
    noise = _noise()
    truth = _zipper_target(size)

    cfa = noise.bayer_sample_rgb(truth.astype(np.float32), pattern)
    bilinear = np.asarray(noise.bilinear_demosaic(np.asarray(cfa, np.float32), pattern), np.float64)
    malvar = np.asarray(noise.malvar_demosaic(np.asarray(cfa, np.float32), pattern), np.float64)

    # Interior only: every demosaic method is poorly defined at the border.
    interior = (slice(2, -2), slice(2, -2))
    edges = _edge_mask(truth)[interior]

    def rmse(a, b, m=None):
        d = (a[interior] - b[interior]) ** 2
        return float(np.sqrt(d[m].mean() if m is not None else d.mean()))

    return DemosaicComparison(
        bilinear=bilinear,
        malvar=malvar,
        difference=np.abs(bilinear - malvar).mean(axis=2),
        bilinear_rmse=rmse(bilinear, truth),
        malvar_rmse=rmse(malvar, truth),
        edge_bilinear_rmse=rmse(bilinear, truth, edges),
        edge_malvar_rmse=rmse(malvar, truth, edges),
    )


def _zipper_target(size: int) -> np.ndarray:
    """Fine luminance detail at slowly varying hue: the zipper-artefact regime.

    Converging bars sweep spatial frequency up to Nyquist, a diagonal edge
    catches directional interpolation errors, and the hue drifts slowly across
    the frame so the image keeps the chroma smoothness real scenes have.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    u, v = x / (size - 1), y / (size - 1)

    # Converging bars: period shrinks toward the right edge, down to 2 px.
    phase = 2.0 * np.pi * (u**2) * size / 2.0
    luminance = 0.5 + 0.38 * np.sign(np.sin(phase))

    # A diagonal luminance step across the lower half.
    luminance = np.where((v > 0.55) & (u + v > 1.1), 0.12, luminance)

    # Slowly varying hue, normalised so it carries chroma and not brightness.
    hue = np.stack(
        [
            0.85 + 0.30 * u,
            0.90 + 0.10 * v,
            1.05 - 0.25 * u,
        ],
        axis=2,
    )
    hue = hue / hue.mean(axis=2, keepdims=True)

    return np.clip(luminance[:, :, None] * hue, 0.0, 1.0)


def _edge_mask(img: np.ndarray) -> np.ndarray:
    g = img.mean(axis=2)
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy) > 0.05


# =====================================================================
# Colour accuracy and spectral overlays
# =====================================================================
@dataclass(frozen=True)
class ColourAccuracy:
    delta_e_2000: np.ndarray
    delta_e_76: np.ndarray
    mean_delta_e: float
    max_delta_e: float
    worst_patch: str
    exposure_scale: float
    neutral_cast_rgb: np.ndarray


def colour_accuracy(chart: Chart, result: IspResult) -> ColourAccuracy:
    """Per-patch CIEDE2000 between the pipeline output and the colorimetric truth.

    Exposure is normalised on the neutral patches first. Without that, being a
    third of a stop dark would register as a large colour error on every patch
    and swamp the thing actually being measured -- so this is what chart-based
    colour accuracy means in practice.
    """
    cs = _cs()
    rendered = _patch_means(result.final_linear)
    reference = chart.reference_srgb_linear

    neutrals = cs.COLORCHECKER_NEUTRAL_SLICE
    scale = _neutral_exposure_scale(rendered, reference)

    # How grey the grey patches actually are. A white balance that worked leaves
    # this at (1, 1, 1); gray-world on a chart whose average is not grey does not.
    cast = rendered[neutrals].mean(axis=0)
    cast = cast / max(float(cast.mean()), 1e-12)

    lab_a = cs.xyz_to_lab(cs.srgb_linear_to_xyz(np.clip(rendered * scale, 0.0, None)))
    lab_b = cs.xyz_to_lab(cs.srgb_linear_to_xyz(np.clip(reference, 0.0, None)))

    de = cs.delta_e_2000(lab_a, lab_b)
    worst = int(np.argmax(de))
    return ColourAccuracy(
        delta_e_2000=de,
        delta_e_76=cs.delta_e_76(lab_a, lab_b),
        mean_delta_e=float(de.mean()),
        max_delta_e=float(de[worst]),
        worst_patch=chart.names[worst],
        exposure_scale=scale,
        neutral_cast_rgb=cast,
    )


def _neutral_exposure_scale(rendered: np.ndarray, reference: np.ndarray) -> float:
    """Gain that puts the neutral patches at the reference level.

    Exposure and colour are separate problems, and keeping them separate is what
    lets a colour error be read as a colour error.
    """
    neutrals = _cs().COLORCHECKER_NEUTRAL_SLICE
    got = float(np.asarray(rendered)[neutrals].mean())
    return float(np.asarray(reference)[neutrals].mean()) / got if got > 1e-9 else 1.0


def _patch_means(img: np.ndarray, *, patch_px: int = PATCH_PX, gap_px: int = 3) -> np.ndarray:
    out = np.zeros((CHART_ROWS * CHART_COLS, 3), dtype=np.float64)
    inset = max(1, patch_px // 5)
    for i in range(CHART_ROWS * CHART_COLS):
        r, c = divmod(i, CHART_COLS)
        y = gap_px + r * (patch_px + gap_px)
        x = gap_px + c * (patch_px + gap_px)
        block = img[y + inset : y + patch_px - inset, x + inset : x + patch_px - inset]
        out[i] = block.reshape(-1, 3).mean(axis=0)
    return out


@dataclass(frozen=True)
class SpectralOverlay:
    """The three curves whose product decides a patch's colour."""

    wavelength_nm: np.ndarray
    illuminant: np.ndarray
    reflectance: np.ndarray
    qe_rgb: np.ndarray
    product_rgb: np.ndarray
    patch_name: str
    cct_k: float


def spectral_overlay(chart: Chart, patch_index: int = 21) -> SpectralOverlay:
    """Illuminant, reflectance and QE on one axis, normalised for comparison.

    Seeing the product is the point: a channel only responds where all three
    curves overlap, which is why a spiky fluorescent lamp renders some surfaces
    so badly.
    """
    cs = _cs()
    idx = int(np.clip(patch_index, 0, chart.reflectance.shape[0] - 1))
    spd = chart.illuminant / max(float(chart.illuminant.max()), 1e-12)
    refl = chart.reflectance[idx]
    product = chart.qe_rgb * (spd * refl)[None, :]

    return SpectralOverlay(
        wavelength_nm=chart.wavelength_nm,
        illuminant=spd,
        reflectance=refl,
        qe_rgb=chart.qe_rgb,
        product_rgb=product,
        patch_name=chart.names[idx],
        cct_k=float(cs.correlated_colour_temperature(cs.xy_chromaticity(chart.white_point_xyz[None, :])[0])),
    )


def to_display(img: np.ndarray) -> np.ndarray:
    """Clamp an image for texture upload without changing its encoding."""
    return np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0)
