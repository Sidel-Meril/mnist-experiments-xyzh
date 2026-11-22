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

RAW_COLUMNS = ["image_id", "label", "mnist_split", "dataset", "label_name"]

IMAGE_HWC = (28, 28, 1)
IMAGE_CHW = (1, 28, 28)
PIXEL_MAX = 255.0

RANDOM_SEED = 7

DIGIT_NAMES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
FASHION_NAMES = (
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle-boot",
)
LABEL_NAMES = DIGIT_NAMES + FASHION_NAMES
DATASETS = ("digits", "fashion")
