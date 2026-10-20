"""Open Cam Studio backend: catalogue, planning, the job queue and file serving."""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from opencam_gui.studio import catalog  # noqa: E402
from opencam_gui.studio.__main__ import main as studio_main  # noqa: E402
from opencam_gui.studio.runs import RunRequest, plan_steps, shell_script  # noqa: E402
from opencam_gui.studio.server import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    with TestClient(create_app(tmp_path / "runs")) as c:
        yield c


def _wait(client, run_id, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/runs/{run_id}").json()
        if r["status"] not in ("queued", "running"):
            return r
        time.sleep(0.2)
    raise AssertionError(f"run {run_id} did not finish")


def test_catalogue_endpoints(client):
    status = client.get("/api/status").json()
    assert {"pbrt_available", "cpu_count", "runs_dir"} <= status.keys()
    scenes = {s["id"]: s for s in client.get("/api/scenes").json()}
    assert scenes["colorchecker"]["modes"] == ["fast_analytic", "pbrt_accurate"]
    assert scenes["slanted_edge"]["modes"] == ["pbrt_accurate"]
    cams = {c["id"]: c for c in client.get("/api/cameras").json()}
    assert "error" not in cams["iphone_8"]
    assert cams["iphone_8"]["pixel_pitch_um"] == pytest.approx(1.4)
    assert cams["default_rccb"]["cfa"] == "RCCB"
    ill = client.get("/api/illuminants").json()
    assert any(i["id"] == "D65" for i in ill)
    spd = client.get("/api/illuminants/D65/spectrum").json()
    assert len(spd["wavelength_nm"]) == len(spd["value"]) > 10
    assert client.get("/api/illuminants/nope/spectrum").status_code == 404
    assert [p["id"] for p in client.get("/api/quality-presets").json()] == ["preview", "standard", "final"]


def test_plan_matches_core_pipeline_and_validates(client):
    body = {"scene_id": "slanted_edge", "camera_id": "iphone_8", "mode": "pbrt_accurate", "xres": 64, "yres": 48}
    plan = client.post("/api/plan", json=body).json()
    names = [s["name"] for s in plan["steps"]]
    assert names[1] == "PBRT spectral render"
    assert "build_image_quality_targets.py" in plan["script"]
    assert plan["estimate_s"] > 0
    bad = client.post("/api/plan", json={**body, "mode": "fast_analytic"})
    assert bad.status_code == 422
    assert client.post("/api/plan", json={**body, "camera_id": "no_such_camera"}).status_code == 422
    assert client.post("/api/plan", json={**body, "xres": 4}).status_code == 422


def test_shell_script_quotes_and_cds():
    steps = plan_steps(RunRequest(scene_id="colorchecker", camera_id="iphone_8"))
    script = shell_script(steps)
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    assert "spectral_sensor_forward.py" in script


def test_fast_run_end_to_end(client, tmp_path):
    body = {"scene_id": "colorchecker", "camera_id": "iphone_8", "name": "studio test", "xres": 96, "yres": 64}
    created = client.post("/api/runs", json=body)
    assert created.status_code == 201
    run = _wait(client, created.json()["id"])
    assert run["status"] == "succeeded", client.get(f"/api/runs/{run['id']}/log").json()["text"][-2000:]
    assert run["id"].endswith("studio-test-iphone-8")
    assert "preview/noisy_demosaic_rgb8.png" in run["outputs"]
    assert "preview/run_stats.json" in run["outputs"]
    assert all(s["returncode"] == 0 for s in run["steps"])

    png = client.get(f"/api/runs/{run['id']}/files/outputs/preview/noisy_demosaic_rgb8.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    script = client.get(f"/api/runs/{run['id']}/files/run.sh").text
    assert "apply_emva_noise.py" in script
    for bad in ("..%2F..%2F..%2Fetc%2Fpasswd", "%2e%2e/%2e%2e/request.json"):
        assert client.get(f"/api/runs/{run['id']}/files/{bad}").status_code == 404
    log = client.get(f"/api/runs/{run['id']}/log").json()
    assert "succeeded" in log["text"]
    assert client.get(f"/api/runs/{run['id']}/log", params={"offset": log["offset"]}).json()["text"] == ""

    run_dir = tmp_path / "runs" / run["id"]
    assert json.loads((run_dir / "request.json").read_text())["camera_id"] == "iphone_8"
    assert [r["id"] for r in client.get("/api/runs").json()] == [run["id"]]
    assert client.delete(f"/api/runs/{run['id']}").status_code == 204
    assert not run_dir.exists()
    assert client.get(f"/api/runs/{run['id']}").status_code == 404


def test_pbrt_mode_refused_without_pbrt(client, monkeypatch):
    monkeypatch.setattr("opencam_gui.studio.runs.pbrt_available", lambda: False)
    r = client.post("/api/runs", json={"scene_id": "colorchecker", "camera_id": "iphone_8", "mode": "pbrt_accurate"})
    assert r.status_code == 422
    assert "PBRT" in r.json()["detail"]


def test_cancel_queued_run(tmp_path):
    app = create_app(tmp_path / "runs", start_worker=False)
    with TestClient(app) as c:
        run = c.post("/api/runs", json={"scene_id": "colorchecker", "camera_id": "iphone_8"}).json()
        assert run["status"] == "queued" and run["queue_position"] == 1
        assert c.post(f"/api/runs/{run['id']}/cancel").json()["status"] == "cancelled"
        assert app.state.jobs.execute(run["id"]).status == "cancelled"


def test_interrupted_runs_are_marked_on_restart(tmp_path):
    with TestClient(create_app(tmp_path / "runs", start_worker=False)) as c:
        run_id = c.post("/api/runs", json={"scene_id": "colorchecker", "camera_id": "iphone_8"}).json()["id"]
    with TestClient(create_app(tmp_path / "runs", start_worker=False)) as c:
        assert c.get(f"/api/runs/{run_id}").json()["status"] == "interrupted"


def test_learn_topics_have_tutorials(client):
    topics = client.get("/api/learn").json()
    assert len(topics) == 8
    assert all(t["tutorial"] for t in topics)
    assert client.post("/api/learn/nope/launch").status_code == 404


def test_estimate_scales_with_samples():
    a = catalog.estimate_seconds("pbrt_accurate", 320, 240, 32)
    b = catalog.estimate_seconds("pbrt_accurate", 320, 240, 3200)
    assert b > 10 * (a - 8)
    assert catalog.estimate_seconds("fast_analytic", 9999, 9999, 9999) < 30


def test_cli_plan_prints_script(tmp_path, capsys):
    req = tmp_path / "request.json"
    req.write_text(json.dumps({"scene_id": "colorchecker", "camera_id": "iphone_8"}))
    assert studio_main(["plan", str(req)]) == 0
    assert "spectral_sensor_forward.py" in capsys.readouterr().out


def test_spa_fallback_serves_index(client):
    r = client.get("/runs/whatever")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
