"""Single source of truth for where things live inside the project."""
from __future__ import annotations

from pathlib import Path

# .../neural-hydrology-explorer
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
CAMELS_DIR = DATA_DIR / "camels_us"          # NeuralHydrology expects this layout (see data/README.md)
RUNS_DIR = PROJECT_ROOT / "runs"              # NeuralHydrology run directories (ignored by git)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

BASIN_LIST_FILE = DATA_DIR / "basins.txt"     # one USGS gauge id per line


def ensure_dirs() -> None:
    for d in (DATA_DIR, CAMELS_DIR, RUNS_DIR, FIGURE_DIR, METRIC_DIR, PREDICTION_DIR):
        d.mkdir(parents=True, exist_ok=True)
