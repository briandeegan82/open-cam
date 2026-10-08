# Contributing

## Setup

- Create a virtual environment: `python3 -m venv venv`
- Install dependencies: `venv/bin/pip install -r requirements-dev.txt`
  (`requirements.txt` alone is enough to run the pipeline; the dev file adds pytest, ruff and the GUI toolkit).
- Build PBRT per `docs/BUILD_PBRT.txt`.

## Before Opening a PR

These are the same checks CI runs (`.github/workflows/ci.yml`):

- Lint: `venv/bin/ruff check .`
- Pipeline unit + module tests: `venv/bin/python -m pytest tests`
  (`PYTHONPATH=tools:. venv/bin/python -m unittest discover -s tests` also works)
- GUI tests: `cd gui && ../venv/bin/python -m pytest`
- Pipeline dry-run: `venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml --dry-run`
- Real render test (needs `third_party/pbrt-v4/build/pbrt`, see `docs/BUILD_PBRT.txt`; skipped otherwise): `venv/bin/python -m pytest tests/test_pbrt_e2e.py`
- Keep generated artifacts (`out/`, `scenes/generated/`) out of commits.

## Tests

- Unit tests for a tool live in `tests/test_<topic>.py`.
- Module tests that run a tool's `main()` end to end use the synthetic EXR / QE / config
  builders in `tests/synthetic_data.py` and write only to temporary directories.
- When optimising a numerical kernel, keep a straightforward reference implementation in
  the test and assert equivalence (see `tests/test_optimized_kernels.py`).

## Coding Notes

- Prefer camera-model driven configuration over hard-coded paths.
- Keep shell and Python pipeline behavior aligned when adding stages.
- Update `README.md` for any config/schema behavior changes.
