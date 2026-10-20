"""Opt-in two-wheelers: bicycles, motorbikes, mopeds, scooters and kick e-scooters.

The models are Objaverse/Sketchfab CC-BY-4.0 meshes listed in config/highway_assets.yaml. Like the
street furniture they are rendered with their sRGB base-colour textures (no measured spectra). Each
model is turned so that its front faces +z before the placement heading is applied, then scaled so
its length along the front axis is a nominal overall length (typical for its type, not checked
against data sheets).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import street_furniture as sf
from fetch_highway_assets import read_ply_positions

# aid -> (kind, direction of the model's front in prepared coordinates [deg from +z towards +x],
#         nominal overall length [m]). Three meshes are not axis-aligned: their fronts are the
# measured principal horizontal axes (tests/test_two_wheelers.py re-measures them).
MODELS = {
    "bike_roadster_old": ("bicycle", 90.0, 1.80),
    "bike_cruiser_red": ("bicycle", 90.0, 1.80),
    "bike_road": ("bicycle", 90.0, 1.75),
    "bike_beach_cruiser": ("bicycle", 0.0, 1.80),
    "bike_family_basket": ("bicycle", 90.0, 1.80),
    "bike_ardis_verona": ("bicycle", 180.0, 1.85),
    "bike_peugeot_retro": ("bicycle", 90.0, 1.80),
    "bike_bmx": ("bicycle", 90.0, 1.60),
    "moto_sarolea_vintage": ("motorcycle", -40.3, 2.00),
    "moped_puch_1978": ("moped", 0.0, 1.75),
    "moto_yamaha_fz150": ("motorcycle", -78.0, 2.00),
    "moto_80s": ("motorcycle", -90.0, 2.10),
    "moto_cruiser": ("motorcycle", 0.0, 2.30),
    "scooter_vespa_yellow": ("scooter", 0.0, 1.80),
    "scooter_vespa_orange": ("scooter", -57.4, 1.80),
    "scooter_honda_squash": ("scooter", 90.0, 1.60),
    "scooter_vespa_ss180": ("scooter", 90.0, 1.80),
    "scooter_vespa_50_special": ("scooter", 0.0, 1.70),
    "moped_solifer_export": ("moped", 180.0, 1.80),
    "scooter_vespa_200_rally": ("scooter", 180.0, 1.80),
    "kickscooter_electric": ("kickscooter", -90.0, 1.08),
}
BICYCLES = tuple(a for a, m in MODELS.items() if m[0] == "bicycle")
POWERED = tuple(a for a, m in MODELS.items() if m[0] in ("motorcycle", "moped", "scooter"))
KICK = tuple(a for a, m in MODELS.items() if m[0] == "kickscooter")


def rot_y(deg: float) -> np.ndarray:
    """pbrt `Rotate deg 0 1 0`: maps +z to (sin deg, 0, cos deg)."""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def placement(aid: str, pts: np.ndarray) -> dict:
    """Yaw that turns the front to +z, scale to the nominal length, base-centre origin, front-aligned bounds."""
    yaw0 = -MODELS[aid][1]
    q = pts @ rot_y(yaw0).T
    lo, hi = np.percentile(q, 0.05, 0), np.percentile(q, 99.95, 0)
    c = np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])
    return {
        "yaw0": yaw0,
        "scale": MODELS[aid][2] / float(hi[2] - lo[2]),
        "origin": c @ rot_y(yaw0),
        "c": c,
        "lo": lo,
        "hi": hi,
    }


class Placer:
    """Defines each model once (ObjectBegin) and instances it; collects pbrt lines and ground truth."""

    def __init__(self, asset, out_dir: Path) -> None:
        self.asset, self.out_dir = asset, out_dir
        self.lines: list[str] = []
        self.meta: list[dict] = []
        self.defined: dict[str, dict | None] = {}

    def variant(self, aid: str) -> dict | None:
        if aid not in self.defined:
            self.defined[aid] = None
            a = self.asset(aid)
            if a is not None:
                info, raw = a
                prep = raw.parent.parent / "prepared" / aid
                v = sf.variants(aid, info, prep)[0]
                pts = np.concatenate([read_ply_positions(prep / info["shapes"][k]["ply"]) for k in v["shapes"]])
                self.lines += sf.object_lines(v["name"], info, prep, raw, self.out_dir, v["shapes"])
                self.defined[aid] = {"name": v["name"], **placement(aid, pts[np.isfinite(pts).all(1)])}
        return self.defined[aid]

    def place(self, aid: str, pos, heading_deg: float, role: str) -> dict | None:
        """Place `aid` with its base centre at `pos`, front towards heading (0 = +z, 90 = +x)."""
        v = self.variant(aid)
        if v is None:
            return None
        s = v["scale"]
        self.lines += sf.instance_lines(v["name"], pos, heading_deg + v["yaw0"], s, v["origin"])
        m = {
            "asset": aid,
            "kind": MODELS[aid][0],
            "role": role,
            "position": [round(float(c), 3) for c in pos],
            "heading_deg": round(float(heading_deg), 2),
            "length_m": MODELS[aid][2],
            "scale": round(s, 5),
            "box_world": sf.world_box(v, pos, heading_deg, s, v["c"]),
        }
        self.meta.append(m)
        return m
