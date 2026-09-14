import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OPENCAM_REPO_ROOT", str(_REPO_ROOT))

_GUI_ROOT = Path(__file__).resolve().parents[1]
if str(_GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(_GUI_ROOT))
