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
PIXEL_MEAN = 0.5
PIXEL_STD = 0.5

# Bottleneck widths compared by the Optuna study. 784 / size is the compression ratio.
LATENT_SIZES = (2, 4, 8, 16, 32, 64, 128)

RANDOM_SEED = 7
