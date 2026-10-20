"""FastAPI app for Open Cam Studio (``opencam-studio`` serves it; see ``gui/README.md``)."""

from __future__ import annotations

import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from opencam_gui.core.repo import python_executable, repo_root
from opencam_gui.studio import catalog
from opencam_gui.studio.runs import JobManager, RunRecord, RunRequest, RunStore, plan_steps, shell_script

STATIC_DIR = Path(__file__).resolve().parent / "static"


class RunRequestBody(BaseModel):
    scene_id: str
    camera_id: str
    mode: Literal["fast_analytic", "pbrt_accurate"] = "fast_analytic"
    illuminant_id: str | None = "D65"
    target_illuminance_lux: float = Field(500.0, gt=0, le=1e6)
    exposure_time_s: float | None = Field(None, gt=0, le=60)
    seed: int = 0
    xres: int = Field(480, ge=16, le=8192)
    yres: int = Field(320, ge=16, le=8192)
    pixelsamples: int = Field(64, ge=1, le=65536)
    name: str = Field("", max_length=80)

    def to_request(self) -> RunRequest:
        return RunRequest(**self.model_dump())


def _run_json(rec: RunRecord, jobs: JobManager) -> dict:
    d = rec.to_dict()
    d["queue_position"] = jobs.queue_position(rec.id) if rec.status == "queued" else None
    req = rec.request
    d["estimate_s"] = catalog.estimate_seconds(req["mode"], req["xres"], req["yres"], req["pixelsamples"])
    return d


def create_app(runs_dir: Path | None = None, *, start_worker: bool = True) -> FastAPI:
    store = RunStore(runs_dir)
    store.mark_interrupted()
    jobs = JobManager(store)
    if start_worker:
        jobs.start()

    app = FastAPI(title="Open Cam Studio", version="0.1.0")
    app.state.store = store
    app.state.jobs = jobs

    def _load(run_id: str) -> RunRecord:
        try:
            return store.load(run_id)
        except KeyError:
            raise HTTPException(404, f"no such run: {run_id}") from None

    # ---- catalogue

    @app.get("/api/status")
    def status() -> dict:
        s = catalog.status()
        s["runs_dir"] = str(store.root)
        return s

    @app.get("/api/scenes")
    def scenes() -> list[dict]:
        return catalog.scenes()

    @app.get("/api/cameras")
    def cameras() -> list[dict]:
        return list(catalog.cameras())

    @app.get("/api/illuminants")
    def illuminants() -> list[dict]:
        return catalog.illuminants()

    @app.get("/api/illuminants/{illuminant_id}/spectrum")
    def illuminant_spectrum(illuminant_id: str) -> dict:
        try:
            return catalog.illuminant_spectrum(illuminant_id)
        except KeyError:
            raise HTTPException(404, f"no such illuminant: {illuminant_id}") from None

    @app.get("/api/quality-presets")
    def quality_presets() -> list[dict]:
        return catalog.quality_presets()

    # ---- planning and runs

    @app.post("/api/plan")
    def plan(body: RunRequestBody) -> dict:
        req = body.to_request()
        try:
            steps = plan_steps(req)
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None
        return {
            "steps": [{"name": s.name, "argv": s.argv} for s in steps],
            "script": shell_script(steps),
            "request": asdict(req),
            "estimate_s": catalog.estimate_seconds(req.mode, req.xres, req.yres, req.pixelsamples),
        }

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return [_run_json(r, jobs) for r in store.list()]

    @app.post("/api/runs", status_code=201)
    def create_run(body: RunRequestBody) -> dict:
        try:
            rec = jobs.submit(body.to_request())
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from None
        return _run_json(rec, jobs)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return _run_json(_load(run_id), jobs)

    @app.get("/api/runs/{run_id}/log")
    def get_log(run_id: str, offset: int = 0) -> dict:
        rec = _load(run_id)
        p = store.log_path(run_id)
        text = ""
        size = 0
        if p.is_file():
            with p.open("rb") as fh:
                fh.seek(max(0, offset))
                data = fh.read(1 << 20)
            text = data.decode("utf-8", errors="replace")
            size = max(0, offset) + len(data)
        return {"text": text, "offset": size or offset, "status": rec.status}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict:
        _load(run_id)
        return _run_json(jobs.cancel(run_id), jobs)

    @app.delete("/api/runs/{run_id}", status_code=204)
    def delete_run(run_id: str) -> None:
        rec = _load(run_id)
        if rec.status in ("queued", "running"):
            raise HTTPException(409, "cancel the run before deleting it")
        store.delete(run_id)

    @app.get("/api/runs/{run_id}/files/{path:path}")
    def run_file(run_id: str, path: str) -> FileResponse:
        _load(run_id)
        base = store.run_dir(run_id).resolve()
        target = (base / path).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(target, filename=target.name if target.suffix not in (".png", ".json") else None)

    # ---- learning demos (Dear PyGui, launched on the machine running the server)

    @app.get("/api/learn")
    def learn_topics() -> list[dict]:
        from opencam_gui.ui.cli import TOPICS

        return [
            {
                "id": t.id,
                "title": t.title,
                "command": f"opencam-gui demo {t.command_names[0]}",
                "tutorial": _tutorial_name(t.id),
            }
            for t in TOPICS
        ]

    @app.post("/api/learn/{topic_id}/launch")
    def launch_demo(topic_id: str) -> dict:
        from opencam_gui.ui.cli import TOPICS

        topic = next((t for t in TOPICS if t.id == topic_id), None)
        if topic is None:
            raise HTTPException(404, f"no such demo: {topic_id}")
        gui_dir = repo_root() / "gui"
        subprocess.Popen(
            [python_executable(), "-m", "opencam_gui.ui.cli", "demo", topic.command_names[0]],
            cwd=str(gui_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return {"launched": topic.id}

    @app.get("/api/learn/{topic_id}/tutorial")
    def tutorial(topic_id: str) -> dict:
        name = _tutorial_name(topic_id)
        if not name:
            raise HTTPException(404, "no tutorial")
        return {"markdown": (repo_root() / "gui" / "tutorials" / name).read_text()}

    # ---- front end

    if (STATIC_DIR / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            candidate = (STATIC_DIR / full_path).resolve()
            if full_path and candidate.is_relative_to(STATIC_DIR) and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(STATIC_DIR / "index.html")
    else:

        @app.get("/", include_in_schema=False)
        def no_frontend() -> HTMLResponse:
            return HTMLResponse(
                "<h1>Open Cam Studio</h1><p>The web front end has not been built. Run "
                "<code>npm ci &amp;&amp; npm run build</code> in <code>gui/studio_web</code>. "
                "The API is live at <a href='/docs'>/docs</a>.</p>"
            )

    return app


_TUTORIALS = {
    "geometry": "01-fundamental-optics.md",
    "optics": "02-psf-aberrations.md",
    "mtf": "03-resolution-mtf.md",
    "sensor": "04-sensor-noise.md",
    "exposure": "05-exposure-defects.md",
    "isp": "06-colour-isp.md",
    "image_generation": "07-image-generation.md",
    "iq_lab": "08-iq-lab.md",
}


def _tutorial_name(topic_id: str) -> str | None:
    name = _TUTORIALS.get(topic_id)
    return name if name and (repo_root() / "gui" / "tutorials" / name).is_file() else None
