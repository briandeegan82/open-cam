# Contributing

## Setup

- Create a virtual environment: `python3 -m venv venv`
- Install dependencies: `venv/bin/pip install -r requirements.txt`
- Build PBRT per `docs/BUILD_PBRT.txt`.

## Before Opening a PR

- Run unit tests: `PYTHONPATH=tools:. venv/bin/python -m unittest discover -s tests -v`
- Run pipeline dry-run: `venv/bin/python tools/run_pipeline.py --config config/pipeline.yaml --dry-run`
- Keep generated artifacts (`out/`, `scenes/generated/`) out of commits.

## Coding Notes

- Prefer camera-model driven configuration over hard-coded paths.
- Keep shell and Python pipeline behavior aligned when adding stages.
- Update `README.md` for any config/schema behavior changes.
