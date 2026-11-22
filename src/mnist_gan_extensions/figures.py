"""PNG helpers shared by train and eval.

``save_grid`` follows the grayscale grid in the RealNVP scaffold: one square
of axes, gray colormap, values in ``[0, 1]``.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mnist_gan_extensions.config import PIXEL_MEAN, PIXEL_STD


def unit_interval(images: np.ndarray | torch.Tensor) -> np.ndarray:
    """Map NCHW pixels from ``[-1, 1]`` to ``[0, 1]``."""
    array = (
        images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    )
    scaled = array.astype(np.float32) * np.float32(PIXEL_STD) + np.float32(PIXEL_MEAN)
    return np.clip(scaled, 0.0, 1.0)


def save_class_grid(
    images: np.ndarray | torch.Tensor,
    path: Path,
    title: str,
    *,
    n_classes: int,
) -> None:
    """One row per class. ``images`` is class-major, ``n_classes`` rows."""
    plt.switch_backend("Agg")
    array = (
        images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    )
    if array.ndim == 4:
        array = array[:, 0]
    count = int(array.shape[0])
    if n_classes < 1 or count < n_classes or count % n_classes != 0:
        raise ValueError("class grid needs a positive multiple of n_classes")
    per_class = count // n_classes
    fig, axes = plt.subplots(
        n_classes,
        per_class,
        figsize=(per_class * 1.1, n_classes * 1.1),
        squeeze=False,
    )
    fig.suptitle(title)
    for index, axis in enumerate(axes.ravel()):
        axis.axis("off")
        axis.imshow(array[index], cmap="gray", vmin=0, vmax=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_grid(images: np.ndarray | torch.Tensor, path: Path, title: str) -> None:
    """Grayscale grid. ``images`` is ``(N, 1, 28, 28)`` or ``(N, 28, 28)`` in ``[0, 1]``."""
    plt.switch_backend("Agg")
    array = (
        images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    )
    if array.ndim == 4:
        array = array[:, 0]
    count = int(array.shape[0])
    if count < 1:
        raise ValueError("image grid needs at least one image")
    side = int(math.ceil(math.sqrt(count)))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.1, side * 1.1), squeeze=False)
    fig.suptitle(title)
    for index, axis in enumerate(axes.ravel()):
        axis.axis("off")
        if index < count:
            axis.imshow(array[index], cmap="gray", vmin=0, vmax=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
