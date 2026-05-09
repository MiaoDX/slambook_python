"""OpenCV mask helpers."""

from __future__ import annotations

import numpy as np


def normalize_mask(mask: np.ndarray | None, *, expected_length: int | None = None) -> np.ndarray:
    """Return a flat boolean inlier mask from an OpenCV mask.

    OpenCV functions are inconsistent about inlier values: some return `0/1`
    masks and others return `0/255`. This helper treats every non-zero value as
    an inlier.
    """

    if mask is None:
        if expected_length is None:
            raise ValueError("expected_length is required when mask is None")
        return np.ones(expected_length, dtype=bool)

    normalized = np.asarray(mask).ravel() != 0
    if expected_length is not None and normalized.size != expected_length:
        raise ValueError(f"mask length {normalized.size} does not match expected length {expected_length}")
    return normalized
