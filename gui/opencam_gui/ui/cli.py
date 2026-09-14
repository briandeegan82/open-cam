"""Command-line entry point for the Open Cam GUI/tutorial system."""

from __future__ import annotations

import argparse
import sys

TOPIC_ALIASES = {
    "optics": "optics",
    "psf": "optics",
    "lens": "optics",
    "sensor": "sensor",
    "sensor-modeling": "sensor",
    "sensor_modeling": "sensor",
    "emva": "sensor",
    "noise": "sensor",
    "image-generation": "image_generation",
    "image_generation": "image_generation",
    "imagegen": "image_generation",
    "pipeline": "image_generation",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="opencam-gui",
        description="Interactive Open Cam demos (Dear PyGui): optics, sensor modelling, image generation.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Launch an interactive demo")
    demo.add_argument("topic", choices=sorted(set(TOPIC_ALIASES)), help="Topic to launch")
    demo.add_argument("--scenario", default=None, help="Optional scripted lecture scenario id")
    demo.add_argument("--list-scenarios", action="store_true", help="List scenarios for the topic and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command != "demo":
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 1

    topic = TOPIC_ALIASES[args.topic]

    if topic == "optics":
        from opencam_gui.topics.optics.scenarios import list_scenarios
        from opencam_gui.ui.desktop.optics_app import run_app
    elif topic == "sensor":
        from opencam_gui.topics.sensor.scenarios import list_scenarios
        from opencam_gui.ui.desktop.sensor_app import run_app
    elif topic == "image_generation":
        from opencam_gui.topics.image_generation.scenarios import list_scenarios
        from opencam_gui.ui.desktop.image_generation_app import run_app
    else:
        print(f"unsupported topic: {args.topic}", file=sys.stderr)
        return 1

    if args.list_scenarios:
        for sc in list_scenarios():
            print(f"{sc.id:28s}  {sc.title}")
            print(f"  {sc.teaching_point}")
        return 0

    run_app(scenario_id=args.scenario)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
