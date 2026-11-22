from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def project_root() -> Path:
    """Repository root in the source tree, otherwise the current directory."""
    candidate = PACKAGE.parent.parent
    if PACKAGE.parent.name == "src" and (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


ROOT = project_root()

MODELING_CONTRACT_PATH = PACKAGE / "model_contract.yaml"
DATA_CONTRACT_PATH = PACKAGE / "data_contract.yaml"

CACHE = ROOT / ".cache" / "mnist"

RAW_COLUMNS = ["image_id", "label", "mnist_split"]

IMAGE_HWC = (28, 28, 1)
IMAGE_CHW = (1, 28, 28)
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5

# Tutorial noise is the data dimension (2). Image runs search these widths.
NOISE_DIMS = (32, 64, 100)
PREVIEW_COUNT = 64
# Generated batch std below this fraction of the real batch std is mode collapse.
COLLAPSE_STD_RATIO = 0.2

RANDOM_SEED = 7
DATASETS = ("digits", "fashion")
