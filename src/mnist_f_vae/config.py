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

CACHE = ROOT / ".cache" / "vision"

RAW_COLUMNS = ["image_id", "label", "image_split", "dataset"]

IMAGE_HWC = (28, 28, 1)
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5

DATASETS = ("mnist_digits", "fashion_mnist")

# Class names for both datasets. One label map cannot serve both, so stage 02
# picks the row from this table with numpy.select.
LABEL_NAMES = {
    "mnist_digits": (
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
    ),
    "fashion_mnist": (
        "t_shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle_boot",
    ),
}

# Even widths. RealNVP alternates a coordinate mask, so the posterior needs
# at least two latent units. 32 is the first probe because the flow acts on z.
LATENT_SIZES = (32, 16, 8)

N_FLOWS = 4
FLOW_HIDDEN = 256
FLOW_SCALE = 1.0

BETA = 1.0
RANDOM_SEED = 7
