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


# Rider contact points in the placed frame (metres; +z front, +y up, x = 0 on the centre line), read off
# side and top point-cloud profiles of each prepared model: seat (z, y), grips (z, y, half-width),
# feet (z, y, half-width). Bicycles: feet = bottom bracket (pedals on 0.17 m cranks); motorbikes: footpegs;
# scooters and mopeds: floorboard or pedals; kick e-scooter: no seat (None), feet on the deck (standing
# rider). The Yamaha FZ150 and Solifer are modelled leaning ~28 and ~17 deg on their side stands; their
# points are read after placement() has stood them upright (DELEAN; the side stand stays out).
RIDER = {
    "bike_roadster_old": ((-0.33, 0.93), (0.15, 0.97, 0.20), (-0.07, 0.30, 0.11)),
    "bike_cruiser_red": ((-0.25, 0.86), (0.22, 1.02, 0.25), (-0.08, 0.28, 0.11)),
    "bike_road": ((-0.18, 1.07), (0.45, 1.02, 0.21), (-0.05, 0.28, 0.11)),
    "bike_beach_cruiser": ((-0.35, 0.88), (0.05, 1.00, 0.18), (-0.07, 0.30, 0.11)),
    "bike_family_basket": ((-0.33, 0.97), (0.25, 1.08, 0.33), (-0.12, 0.28, 0.11)),
    "bike_ardis_verona": ((-0.32, 0.93), (0.25, 1.12, 0.30), (-0.15, 0.28, 0.11)),
    "bike_peugeot_retro": ((-0.22, 0.87), (0.25, 0.92, 0.20), (-0.05, 0.25, 0.11)),
    "bike_bmx": ((-0.20, 0.72), (0.33, 1.02, 0.40), (-0.10, 0.30, 0.11)),
    "moto_sarolea_vintage": ((-0.35, 0.88), (0.25, 1.00, 0.33), (0.00, 0.25, 0.15)),
    "moped_puch_1978": ((-0.25, 0.80), (0.25, 1.02, 0.33), (-0.05, 0.25, 0.15)),
    "moto_80s": ((-0.30, 0.80), (0.30, 1.05, 0.40), (-0.15, 0.35, 0.17)),
    "moto_cruiser": ((-0.15, 0.75), (0.30, 1.15, 0.33), (0.45, 0.40, 0.20)),
    "scooter_vespa_yellow": ((-0.35, 0.72), (0.25, 1.10, 0.25), (0.05, 0.25, 0.13)),
    "scooter_vespa_orange": ((-0.35, 0.80), (0.35, 1.05, 0.30), (0.00, 0.30, 0.13)),
    "scooter_honda_squash": ((-0.40, 0.75), (0.22, 1.12, 0.42), (-0.05, 0.30, 0.13)),
    "scooter_vespa_ss180": ((-0.35, 0.73), (0.22, 1.05, 0.28), (0.00, 0.25, 0.13)),
    "scooter_vespa_50_special": ((-0.35, 0.78), (0.25, 1.00, 0.33), (-0.05, 0.25, 0.13)),
    "scooter_vespa_200_rally": ((-0.35, 0.75), (0.40, 1.05, 0.32), (0.05, 0.25, 0.13)),
    "moto_yamaha_fz150": ((-0.20, 0.83), (0.35, 1.02, 0.33), (-0.15, 0.33, 0.17)),
    "moped_solifer_export": ((-0.30, 0.70), (0.50, 0.93, 0.27), (-0.10, 0.25, 0.15)),
    "kickscooter_electric": (None, (0.22, 1.13, 0.22), (-0.15, 0.12, 0.05)),
}


def rot_y(deg: float) -> np.ndarray:
    """pbrt `Rotate deg 0 1 0`: maps +z to (sin deg, 0, cos deg)."""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])


def rot_z(deg: float) -> np.ndarray:
    """pbrt `Rotate deg 0 0 1`: maps +y to (-sin deg, cos deg, 0)."""
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


#: models stored leaning on their side stands; placement() stands them upright
DELEAN = ("moto_yamaha_fz150", "moped_solifer_export")


def side_lean_deg(q: np.ndarray, iters: int = 6) -> float:
    """Roll [deg, top towards +x positive] that makes front-aligned points ``q`` upright: the slope of x
    against height over the points above 15 % of the height, refitted after each correction."""
    roll = 0.0
    for _ in range(iters):
        r = q @ rot_z(roll).T
        lo, hi = np.percentile(r[:, 1], 0.05), np.percentile(r[:, 1], 99.95)
        m = r[:, 1] > lo + 0.15 * (hi - lo)
        roll += math.degrees(math.atan(np.polyfit(r[m, 1], r[m, 0], 1)[0]))
    return roll


def placement(aid: str, pts: np.ndarray, upright: bool = False) -> dict:
    """Yaw that turns the front to +z, scale to the nominal length, base-centre origin, front-aligned bounds.

    With ``upright``, models in DELEAN are also rolled upright about the travel axis (``roll``, pbrt Rotate about z,
    from side_lean_deg; the side stand stays out)."""
    yaw0 = -MODELS[aid][1]
    q = pts @ rot_y(yaw0).T
    roll = 0.0
    if upright and aid in DELEAN:
        roll = side_lean_deg(q)
        q = q @ rot_z(roll).T
    lo, hi = np.percentile(q, 0.05, 0), np.percentile(q, 99.95, 0)
    c = np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])
    return {
        "yaw0": yaw0,
        "scale": MODELS[aid][2] / float(hi[2] - lo[2]),
        "origin": c @ rot_z(roll) @ rot_y(yaw0),
        "roll": roll,
        "c": c,
        "lo": lo,
        "hi": hi,
    }


def instance_lines(v: dict, pos, heading_deg: float) -> list[str]:
    """Instance a placed variant: base centre at ``pos``, front towards ``heading_deg`` (0 = +z)."""
    if not v.get("roll"):
        return sf.instance_lines(v["name"], pos, heading_deg + v["yaw0"], v["scale"], v["origin"])
    f = sf._f
    return [
        "AttributeBegin",
        f"    Translate {f(pos[0])} {f(pos[1])} {f(pos[2])}",
        f"    Rotate {f(heading_deg)} 0 1 0",
        f"    Rotate {f(v['roll'])} 0 0 1",
        f"    Rotate {f(v['yaw0'])} 0 1 0",
        f"    Scale {f(v['scale'])} {f(v['scale'])} {f(v['scale'])}",
        f"    Translate {f(-v['origin'][0])} {f(-v['origin'][1])} {f(-v['origin'][2])}",
        f'    ObjectInstance "{v["name"]}"',
        "AttributeEnd",
    ]


class Placer:
    """Defines each model once (ObjectBegin) and instances it; collects pbrt lines and ground truth."""

    def __init__(self, asset, out_dir: Path, upright: bool = False) -> None:
        """``upright``: stand side-stand models up (ridden); parked ones keep leaning on the stand."""
        self.asset, self.out_dir, self.upright = asset, out_dir, upright
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
                self.defined[aid] = {"name": v["name"], **placement(aid, pts[np.isfinite(pts).all(1)], self.upright)}
        return self.defined[aid]

    def place(self, aid: str, pos, heading_deg: float, role: str) -> dict | None:
        """Place `aid` with its base centre at `pos`, front towards heading (0 = +z, 90 = +x)."""
        v = self.variant(aid)
        if v is None:
            return None
        s = v["scale"]
        self.lines += instance_lines(v, pos, heading_deg)
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
