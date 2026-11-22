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

# The beta sweep is the comparison, so the latent width stays put.
LATENT_DIM = 8

# KL weights. 1 is the standard VAE. A larger beta pulls q(z|x) toward the prior.
BETAS = (0.1, 1.0, 4.0, 10.0)

# A posterior-mean axis counts as active when its variance across images exceeds this.
ACTIVE_UNIT_THRESHOLD = 0.01

TRAVERSAL_STEPS = 8
TRAVERSAL_SPAN = 3.0
DIVERSITY_SAMPLES = 64

RANDOM_SEED = 7
