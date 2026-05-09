"""Common feature matching result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureMatchSet:
    """Matched keypoints and descriptors from two images."""

    keypoints0: Sequence[object]
    keypoints1: Sequence[object]
    descriptors0: np.ndarray | None
    descriptors1: np.ndarray | None
    matches: Sequence[object]

    @property
    def points0(self) -> np.ndarray:
        points0, _, _ = matches_to_points(self.keypoints0, self.keypoints1, self.matches)
        return points0

    @property
    def points1(self) -> np.ndarray:
        _, points1, _ = matches_to_points(self.keypoints0, self.keypoints1, self.matches)
        return points1

    @property
    def distances(self) -> np.ndarray:
        _, _, distances = matches_to_points(self.keypoints0, self.keypoints1, self.matches)
        return distances


def matches_to_points(
    keypoints0: Sequence[object],
    keypoints1: Sequence[object],
    matches: Sequence[object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert OpenCV DMatch objects into `Nx2` point arrays and distances."""

    points0 = []
    points1 = []
    distances = []

    for match in matches:
        points0.append(keypoints0[match.queryIdx].pt)
        points1.append(keypoints1[match.trainIdx].pt)
        distances.append(match.distance)

    return (
        np.asarray(points0, dtype=np.float32).reshape(-1, 2),
        np.asarray(points1, dtype=np.float32).reshape(-1, 2),
        np.asarray(distances, dtype=np.float32),
    )
