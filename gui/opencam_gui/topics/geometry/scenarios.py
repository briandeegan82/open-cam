"""Scripted lecture scenarios for the fundamental-optics / imaging-geometry demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    teaching_point: str
    notes: str
    focal_length_mm: float
    f_number: float
    focus_distance_mm: float
    format_id: str
    pixel_pitch_um: float
    object_height_mm: float = 1700.0  # a standing person, for the ray diagram
    use_pixel_coc: bool = False
    distortion_k1: float = 0.0
    distortion_k2: float = 0.0
    camera_recipe_id: str | None = None


SCENARIOS: dict[str, Scenario] = {
    "normal_50mm": Scenario(
        id="normal_50mm",
        title="The 'normal' lens: 50 mm on full frame",
        teaching_point=(
            "A focal length near the sensor diagonal gives roughly 47 degrees diagonally  -  the "
            "reference every other focal length is described relative to."
        ),
        notes=(
            "Read the field of view, then change only the sensor format and watch the same "
            "50 mm lens become a telephoto."
        ),
        focal_length_mm=50.0,
        f_number=5.6,
        focus_distance_mm=3000.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        camera_recipe_id="nikon_z6",
    ),
    "phone_wide": Scenario(
        id="phone_wide",
        title="Phone wide: short lens, tiny sensor, deep focus",
        teaching_point=(
            "A 4 mm lens on a 1/2.55-inch sensor has the same field of view as a 26 mm full-frame "
            "lens, but the hyperfocal distance collapses to centimetres  -  which is why phone "
            "photos are sharp everywhere and hard to blur the background with."
        ),
        notes="Compare the hyperfocal distance against the 50 mm full-frame scenario.",
        focal_length_mm=4.3,
        f_number=1.8,
        focus_distance_mm=1000.0,
        format_id="phone_1_2_55",
        pixel_pitch_um=1.22,
        object_height_mm=300.0,
        camera_recipe_id="iphone_8",
    ),
    "portrait_shallow_dof": Scenario(
        id="portrait_shallow_dof",
        title="Portrait: long lens, wide aperture, shallow depth of field",
        teaching_point=(
            "85 mm at f/1.8 on a head-and-shoulders subject puts the whole depth of field inside "
            "a few centimetres  -  the eyes can be sharp while the ears are not."
        ),
        notes="Drag the f-number and watch the near/far limits close in around the focus plane.",
        focal_length_mm=85.0,
        f_number=1.8,
        focus_distance_mm=1500.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        object_height_mm=400.0,
        camera_recipe_id="nikon_z6",
    ),
    "landscape_hyperfocal": Scenario(
        id="landscape_hyperfocal",
        title="Landscape: hyperfocal focusing and the diffraction wall",
        teaching_point=(
            "Focus at the hyperfocal distance and everything from H/2 to infinity is acceptably "
            "sharp. But stopping down further to extend it runs into diffraction: the aperture "
            "trade-off plot has a minimum, and past it sharpness gets worse, not better."
        ),
        notes="Find the optimum f-number on the aperture plot, then compare it against f/22.",
        focal_length_mm=24.0,
        f_number=11.0,
        focus_distance_mm=5000.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        object_height_mm=2000.0,
        camera_recipe_id="nikon_z6",
    ),
    "telephoto_250mm": Scenario(
        id="telephoto_250mm",
        title="Telephoto 250 mm: narrow field, compressed perspective",
        teaching_point=(
            "A long lens crops the field to under 10 degrees. To frame the same subject you stand "
            "much further back, which is what flattens ('compresses') the perspective  -  the "
            "focal length changes the framing, the distance changes the perspective."
        ),
        notes="Compare the subject footprint against the 50 mm scenario at the same distance.",
        focal_length_mm=250.0,
        f_number=5.6,
        focus_distance_mm=20000.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        object_height_mm=1700.0,
        camera_recipe_id="nikon_z6",
    ),
    "macro_life_size": Scenario(
        id="macro_life_size",
        title="Macro at 1:1: magnification costs two stops",
        teaching_point=(
            "At life size the image distance equals the object distance, |m| = 1, and the "
            "(1 + m)^2 bellows factor costs two stops. The working f-number is double the marked "
            "one, so f/2.8 behaves like f/5.6  -  the same term the sensor-forward stage applies."
        ),
        notes="Watch the ray diagram: this is the only scenario where object and image are symmetric.",
        focal_length_mm=100.0,
        f_number=2.8,
        focus_distance_mm=200.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        object_height_mm=36.0,
        use_pixel_coc=True,
        camera_recipe_id="nikon_z6",
    ),
    "barrel_distortion": Scenario(
        id="barrel_distortion",
        title="Wide-angle barrel distortion",
        teaching_point=(
            "An uncorrected short lens maps straight lines to curves, bowing them outward "
            "(k1 < 0 = barrel). This is the same Brown-Conrady model the analytic sensor-forward "
            "path inverts to map pixels back to scene coordinates."
        ),
        notes="Flip k1 positive to get pincushion, and read the corner distortion percentage.",
        focal_length_mm=14.0,
        f_number=4.0,
        focus_distance_mm=2000.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        object_height_mm=1700.0,
        distortion_k1=-0.18,
        distortion_k2=0.04,
    ),
    "corner_falloff": Scenario(
        id="corner_falloff",
        title="Natural vignetting: cos^4 corner falloff",
        teaching_point=(
            "Even a mechanically perfect lens loses corner light to the cos^4 of the chief-ray "
            "angle. On a wide lens over a large sensor that is more than a stop before any "
            "mechanical vignetting is considered."
        ),
        notes="Sweep the focal length from 14 mm to 85 mm and watch the falloff flatten out.",
        focal_length_mm=16.0,
        f_number=4.0,
        focus_distance_mm=4000.0,
        format_id="full_frame",
        pixel_pitch_um=5.94,
        object_height_mm=1700.0,
        camera_recipe_id="nikon_z6",
    ),
    "equivalence": Scenario(
        id="equivalence",
        title="Equivalence: same framing, different depth of field",
        teaching_point=(
            "Micro Four Thirds at 25 mm f/2.8 frames exactly like full frame at 50 mm f/2.8, but "
            "the physical aperture is half the diameter, so the depth of field is twice as deep. "
            "Crop factor scales focal length and the depth of field, not the f-number."
        ),
        notes="Switch the format between Micro Four Thirds and full frame with the focal length halved/doubled.",
        focal_length_mm=25.0,
        f_number=2.8,
        focus_distance_mm=2000.0,
        format_id="micro_four_thirds",
        pixel_pitch_um=3.3,
        object_height_mm=1700.0,
    ),
}


def list_scenarios() -> list[Scenario]:
    return list(SCENARIOS.values())


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        known = ", ".join(SCENARIOS)
        raise KeyError(f"unknown scenario {scenario_id!r}; choose from: {known}") from exc
