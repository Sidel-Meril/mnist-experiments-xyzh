"""Domain KID between images and prior samples.

The score is the unbiased degree-3 polynomial MMD. Features are non-overlapping
block means at widths 2, 4, and 7, plus the image mean and standard deviation.
Both sets are centered and scaled with the real-image statistics. A lower
score means the prior samples sit closer to those images.
"""

from __future__ import annotations

import numpy as np


def block_means(gray: np.ndarray, block: int) -> np.ndarray:
    count, height, width = gray.shape
    pooled = gray.reshape(count, height // block, block, width // block, block).mean(axis=(2, 4))
    return pooled.reshape(count, -1)


def spatial_features(images: np.ndarray) -> np.ndarray:
    """Block means at widths 2, 4, and 7, plus the image mean and standard deviation."""
    gray = np.asarray(images[:, 0], dtype=np.float64)
    parts = [block_means(gray, block) for block in (2, 4, 7)]
    parts.append(gray.mean(axis=(1, 2))[:, None])
    parts.append(gray.std(axis=(1, 2))[:, None])
    return np.concatenate(parts, axis=1)


def polynomial_mmd(real: np.ndarray, fake: np.ndarray) -> float:
    """Unbiased degree-3 polynomial MMD. The kernel is ``(x @ y / dim + 1) ** 3``."""
    width = real.shape[1]
    real_kernel = (real @ real.T / width + 1.0) ** 3
    fake_kernel = (fake @ fake.T / width + 1.0) ** 3
    cross_kernel = (real @ fake.T / width + 1.0) ** 3
    n_real = real.shape[0]
    n_fake = fake.shape[0]
    real_term = (real_kernel.sum() - np.trace(real_kernel)) / (n_real * (n_real - 1))
    fake_term = (fake_kernel.sum() - np.trace(fake_kernel)) / (n_fake * (n_fake - 1))
    return float(real_term + fake_term - 2.0 * cross_kernel.mean())


def domain_kid(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """Kernel distance between real images and prior samples. Lower is closer."""
    real_features = spatial_features(real_images)
    fake_features = spatial_features(fake_images)
    center = real_features.mean(axis=0)
    scale = real_features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return polynomial_mmd((real_features - center) / scale, (fake_features - center) / scale)
