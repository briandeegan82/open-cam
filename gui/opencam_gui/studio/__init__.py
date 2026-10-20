"""Open Cam Studio: a local web interface over the Open Cam pipeline.

The FastAPI backend (:mod:`.server`) serves a React front end (built into
``static/`` from ``gui/studio_web``) and runs the same ``tools/*.py`` command
plans as the CLI and the Dear PyGui demos (:mod:`opencam_gui.core.pipeline`).
Every run is saved under the runs directory with its request, the exact shell
commands (``run.sh``), the log and the outputs, so GUI runs stay scriptable.
"""
