"""Command-line entry point for the Open Cam GUI/tutorial system.

Demos are declared once in :data:`TOPICS`. Adding a new one is a single entry
here plus the ``topics/<id>/scenarios.py`` and ``ui/desktop/<id>_app.py``
modules it names -- nothing else in the CLI needs to change.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    aliases: tuple[str, ...]
    scenarios_module: str
    app_module: str

    @property
    def command_names(self) -> tuple[str, ...]:
        return (self.id.replace("_", "-"), *self.aliases)

    def list_scenarios(self):
        return import_module(self.scenarios_module).list_scenarios()

    def run_app(self, scenario_id: str | None) -> None:
        import_module(self.app_module).run_app(scenario_id=scenario_id)


TOPICS: tuple[Topic, ...] = (
    Topic(
        id="geometry",
        title="01 - Fundamental optics / imaging geometry",
        aliases=("fundamentals", "imaging-geometry", "lens-geometry", "dof"),
        scenarios_module="opencam_gui.topics.geometry.scenarios",
        app_module="opencam_gui.ui.desktop.geometry_app",
    ),
    Topic(
        id="optics",
        title="02 - PSF, diffraction and aberrations",
        aliases=("psf", "lens", "aberrations"),
        scenarios_module="opencam_gui.topics.optics.scenarios",
        app_module="opencam_gui.ui.desktop.optics_app",
    ),
    Topic(
        id="mtf",
        title="03 - Resolution and MTF",
        aliases=("resolution", "sfr", "slanted-edge"),
        scenarios_module="opencam_gui.topics.mtf.scenarios",
        app_module="opencam_gui.ui.desktop.mtf_app",
    ),
    Topic(
        id="sensor",
        title="04 - Sensor noise modelling (EMVA1288)",
        aliases=("sensor-modeling", "sensor_modeling", "emva", "noise"),
        scenarios_module="opencam_gui.topics.sensor.scenarios",
        app_module="opencam_gui.ui.desktop.sensor_app",
    ),
    Topic(
        id="exposure",
        title="05 - Exposure and sensor defects",
        aliases=("defects", "iso", "exposure-triangle"),
        scenarios_module="opencam_gui.topics.exposure.scenarios",
        app_module="opencam_gui.ui.desktop.exposure_app",
    ),
    Topic(
        id="isp",
        title="06 - Colour and the ISP",
        aliases=("colour", "color", "demosaic", "cfa"),
        scenarios_module="opencam_gui.topics.isp.scenarios",
        app_module="opencam_gui.ui.desktop.isp_app",
    ),
    Topic(
        id="image_generation",
        title="07 - Image generation (scene to image)",
        aliases=("imagegen", "pipeline", "end-to-end"),
        scenarios_module="opencam_gui.topics.image_generation.scenarios",
        app_module="opencam_gui.ui.desktop.image_generation_app",
    ),
)

TOPIC_BY_NAME: dict[str, Topic] = {
    name: topic for topic in TOPICS for name in topic.command_names
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opencam-gui",
        description="Interactive Open Cam camera-simulation demos (Dear PyGui).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Launch an interactive demo")
    demo.add_argument("topic", choices=sorted(TOPIC_BY_NAME), help="Topic to launch")
    demo.add_argument("--scenario", default=None, help="Optional scripted lecture scenario id")
    demo.add_argument("--list-scenarios", action="store_true", help="List scenarios for the topic and exit")

    sub.add_parser("topics", help="List the demos in teaching order and exit")
    return p


def _print_topics() -> None:
    print("Open Cam demos, in the order the tutorials build on each other:\n")
    for topic in TOPICS:
        print(f"  {topic.title}")
        print(f"      opencam-gui demo {topic.id.replace('_', '-')}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "topics":
        _print_topics()
        return 0

    if args.command != "demo":
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 1

    topic = TOPIC_BY_NAME[args.topic]

    if args.list_scenarios:
        for sc in topic.list_scenarios():
            print(f"{sc.id:28s}  {sc.title}")
            print(f"  {sc.teaching_point}")
        return 0

    topic.run_app(args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
