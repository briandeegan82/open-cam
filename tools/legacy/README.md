# Legacy one-off tools

Migration scripts kept for provenance (`migrated_from:` keys in `config/camera_recipes/*.yaml` point back to
the files they produced). Nothing in `tools/`, `gui/`, `scripts/`, `tests/` or CI calls them; they operate on
the legacy `config/camera_models/` tree, which `load_camera_model` refuses unless explicitly allowed.

| script | what it did |
|---|---|
| `split_camera_models.py` | split full `config/camera_models/*.yaml` into recipes + lens/sensor models |
| `fix_qe_import_all_models.py` | re-imported QE curves for every legacy model |
| `audit_qe_import_health.py` | reported QE-import problems across legacy models |
| `assign_lens_models.py` | assigned a lens model to every camera (see docs/LENS_MODELS.txt) |

Run from the repo root, e.g. `venv/bin/python tools/legacy/assign_lens_models.py --dry-run`.
