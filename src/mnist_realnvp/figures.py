"""PNG helpers shared by train and eval."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_curve(train: list[float], test: list[float], plot_path: Path, ylabel: str) -> None:
    plt.switch_backend("Agg")
    fig, ax = plt.subplots()
    ax.plot(train, label="train")
    ax.plot(test, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_grid(images: np.ndarray | torch.Tensor, path: Path, title: str) -> None:
    """Grayscale grid. ``images`` is ``(N, 1, 28, 28)`` or ``(N, 28, 28)``."""
    plt.switch_backend("Agg")
    array = (
        images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    )
    if array.ndim == 4:
        array = array[:, 0]
    count = int(array.shape[0])
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


def save_reconstructions(
    original: np.ndarray | torch.Tensor,
    restored: np.ndarray | torch.Tensor,
    path: Path,
    title: str,
) -> None:
    """Two rows: data on top, inverse-flow reconstructions below."""
    plt.switch_backend("Agg")
    left = (
        original.detach().cpu().numpy()
        if isinstance(original, torch.Tensor)
        else np.asarray(original)
    )
    right = (
        restored.detach().cpu().numpy()
        if isinstance(restored, torch.Tensor)
        else np.asarray(restored)
    )
    if left.ndim == 4:
        left = left[:, 0]
    if right.ndim == 4:
        right = right[:, 0]
    count = int(left.shape[0])
    fig, axes = plt.subplots(2, count, figsize=(count * 1.1, 2.4), squeeze=False)
    fig.suptitle(title)
    for index in range(count):
        axes[0, index].imshow(left[index], cmap="gray", vmin=0, vmax=1)
        axes[1, index].imshow(right[index], cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    axes[0, 0].set_title("data", fontsize=8)
    axes[1, 0].set_title("inverse", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
