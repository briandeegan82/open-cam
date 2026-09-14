"""Interactive GUI + tutorials for Open Cam (optics, sensor modelling, image generation).

This package is a thin interface layer on top of the ``tools/`` pipeline
scripts. It does not reimplement any camera physics: every plotted curve or
generated image is produced by importing or subprocess-calling the existing
``tools/*.py`` modules, the same ones ``tools/run_pipeline.py`` uses.
"""

from __future__ import annotations
