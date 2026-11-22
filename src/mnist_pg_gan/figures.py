"""PNG helpers shared by train and eval.

``save_grid`` draws an RGB batch. ``save_growth_rows`` stacks one row per
resolution phase so the same latent codes can be compared as the generator grows.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mnist_pg_gan.config import PIXEL_MEAN, PIXEL_STD


def unit_interval(images: np.ndarray | torch.Tensor) -> np.ndarray:
    """Map NCHW pixels from ``[-1, 1]`` to ``[0, 1]``."""
    array = (
        images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    )
    scaled = array.astype(np.float32) * np.float32(PIXEL_STD) + np.float32(PIXEL_MEAN)
    return np.clip(scaled, 0.0, 1.0)


def _gallery(images: np.ndarray) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim == 4 and array.shape[1] in (1, 3):
        array = np.transpose(array, (0, 2, 3, 1))
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    return array


def _nearest(image: np.ndarray, size: int) -> np.ndarray:
    height = int(image.shape[0])
    if height == size:
        return image
    if height < 1 or size % height != 0:
        raise ValueError(f"cannot nearest-resize {height}px to {size}px")
    factor = size // height
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def save_grid(images: np.ndarray | torch.Tensor, path: Path, title: str) -> None:
    """RGB or gray grid. ``images`` is NCHW or NHW in ``[0, 1]``."""
    plt.switch_backend("Agg")
    array = (
        images.detach().cpu().numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
    )
    array = _gallery(array)
    count = int(array.shape[0])
    if count < 1:
        raise ValueError("image grid needs at least one image")
    side = int(math.ceil(math.sqrt(count)))
    fig, axes = plt.subplots(side, side, figsize=(side * 1.1, side * 1.1), squeeze=False)
    fig.suptitle(title)
    for index, axis in enumerate(axes.ravel()):
        axis.axis("off")
        if index < count:
            axis.imshow(array[index], vmin=0, vmax=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_growth_rows(
    rows: list[tuple[np.ndarray, str]],
    path: Path,
    *,
    display_size: int,
) -> None:
    """One row per growth phase. Tiles are nearest-upsampled to ``display_size``."""
    if not rows:
        raise ValueError("growth timeline needs at least one phase")
    plt.switch_backend("Agg")
    galleries = [_gallery(images) for images, _caption in rows]
    captions = [caption for _images, caption in rows]
    count = min(int(gallery.shape[0]) for gallery in galleries)
    if count < 1:
        raise ValueError("growth timeline needs at least one image")
    fig, axes = plt.subplots(
        len(rows),
        count,
        figsize=(count * 1.15, len(rows) * 1.35),
        squeeze=False,
    )
    for gallery, caption, axis_row in zip(galleries, captions, axes, strict=True):
        for column, axis in enumerate(axis_row):
            axis.axis("off")
            if column >= count:
                continue
            axis.imshow(_nearest(gallery[column], display_size), vmin=0, vmax=1)
            if column == 0:
                axis.set_title(caption, loc="left", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
