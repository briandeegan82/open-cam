"""Opt-in two-wheelers: bicycles, motorbikes, mopeds, scooters and kick e-scooters.

The models are Objaverse/Sketchfab CC-BY-4.0 meshes listed in config/highway_assets.yaml. Like the
street furniture they are rendered with their sRGB base-colour textures (no measured spectra). Each
model is scaled to a nominal overall length (typical for its type, not checked against data sheets)
and turned so that its front faces +z before the placement heading is applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import street_furniture as sf

# aid -> (kind, model axis its front points along, nominal overall length [m])
MODELS = {
    "bike_roadster_old": ("bicycle", "+x", 1.80),
    "bike_cruiser_red": ("bicycle", "+x", 1.80),
    "bike_road": ("bicycle", "+x", 1.75),
    "bike_beach_cruiser": ("bicycle", "+z", 1.80),
    "bike_family_basket": ("bicycle", "+x", 1.80),
    "bike_ardis_verona": ("bicycle", "-z", 1.85),
    "bike_peugeot_retro": ("bicycle", "+x", 1.80),
    "bike_bmx": ("bicycle", "+x", 1.60),
    "moto_sarolea_vintage": ("motorcycle", "+z", 2.00),
    "moped_puch_1978": ("moped", "+z", 1.75),
    "moto_yamaha_fz150": ("motorcycle", "-x", 2.00),
    "moto_80s": ("motorcycle", "-x", 2.10),
    "moto_cruiser": ("motorcycle", "+z", 2.30),
    "scooter_vespa_yellow": ("scooter", "+z", 1.80),
    "scooter_vespa_orange": ("scooter", "-x", 1.80),
    "scooter_honda_squash": ("scooter", "+x", 1.60),
    "scooter_vespa_ss180": ("scooter", "+x", 1.80),
    "scooter_vespa_50_special": ("scooter", "+z", 1.70),
    "moped_solifer_export": ("moped", "-z", 1.80),
    "scooter_vespa_200_rally": ("scooter", "-z", 1.80),
    "kickscooter_electric": ("kickscooter", "-x", 1.08),
}
# pbrt `Rotate a 0 1 0` maps model +z to (sin a, 0, cos a): the yaw that turns each front axis to +z
FRONT_YAW = {"+z": 0.0, "+x": -90.0, "-x": 90.0, "-z": 180.0}
BICYCLES = tuple(a for a, m in MODELS.items() if m[0] == "bicycle")
POWERED = tuple(a for a, m in MODELS.items() if m[0] in ("motorcycle", "moped", "scooter"))
KICK = tuple(a for a, m in MODELS.items() if m[0] == "kickscooter")


def placement(aid: str, v: dict) -> tuple[float, np.ndarray, float]:
    """Scale to the nominal length, base-centre origin and the yaw that turns the front to +z."""
    _, front, length = MODELS[aid]
    ax = 0 if front[1] == "x" else 2
    lo, hi = v["lo"], v["hi"]
    s = length / float(hi[ax] - lo[ax])
    return s, np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])]), FRONT_YAW[front]


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
                self.lines += sf.object_lines(v["name"], info, prep, raw, self.out_dir, v["shapes"])
                self.defined[aid] = v
        return self.defined[aid]

    def place(self, aid: str, pos, heading_deg: float, role: str) -> dict | None:
        """Place `aid` with its base centre at `pos`, front towards heading (0 = +z, 90 = +x)."""
        v = self.variant(aid)
        if v is None:
            return None
        s, origin, yaw0 = placement(aid, v)
        yaw = heading_deg + yaw0
        self.lines += sf.instance_lines(v["name"], pos, yaw, s, origin)
        m = {
            "asset": aid,
            "kind": MODELS[aid][0],
            "role": role,
            "position": [round(float(c), 3) for c in pos],
            "heading_deg": round(float(heading_deg), 2),
            "length_m": MODELS[aid][2],
            "scale": round(s, 5),
            "box_world": sf.world_box(v, pos, yaw, s, origin),
        }
        self.meta.append(m)
        return m
