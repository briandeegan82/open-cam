"""``opencam-studio``: serve the Studio web app, or run/plan a saved request headlessly.

    opencam-studio                          # serve on http://127.0.0.1:8765 and open a browser
    opencam-studio serve --port 9000 --no-browser
    opencam-studio plan  request.json       # print the shell commands a request runs
    opencam-studio run   request.json       # run it without the GUI (saved like a GUI run)

``request.json`` is the file Studio saves in every run directory, so a GUI run can be
repeated or batch-edited from a script.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path


def _load_request(path: Path):
    from opencam_gui.studio.runs import RunRequest

    return RunRequest.from_dict(json.loads(Path(path).read_text()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opencam-studio", description="Open Cam Studio web interface.")
    sub = p.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Serve the web app (default command)")
    serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true", help="Don't open a browser tab")
    serve.add_argument("--runs-dir", type=Path, default=None, help="Run history folder (default out/studio_runs)")

    plan = sub.add_parser("plan", help="Print the shell commands for a saved request.json")
    plan.add_argument("request", type=Path)

    run = sub.add_parser("run", help="Run a saved request.json without the GUI")
    run.add_argument("request", type=Path)
    run.add_argument("--runs-dir", type=Path, default=None)
    return p


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from opencam_gui.studio.server import create_app

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"warning: binding to {args.host} exposes Studio (which launches pipeline jobs) to the network; "
            "prefer an SSH tunnel to 127.0.0.1.",
            file=sys.stderr,
        )
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"Open Cam Studio: {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(args.runs_dir), host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in ("serve", "plan", "run", "-h", "--help"):
        argv = ["serve", *argv]
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)

    from opencam_gui.studio.runs import JobManager, RunStore, plan_steps, shell_script

    req = _load_request(args.request)
    if args.command == "plan":
        print(shell_script(plan_steps(req)))
        return 0
    store = RunStore(args.runs_dir)
    jobs = JobManager(store)
    rec = jobs.submit(req)
    rec = jobs.execute(rec.id, echo=True)
    print(f"run {rec.id}: {rec.status} -> {store.run_dir(rec.id)}")
    return 0 if rec.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
