"""Trajectory import/export helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PoseStamped:
    """Timestamped camera pose as `T_wc`."""

    timestamp: float
    transform_wc: np.ndarray

    def __post_init__(self) -> None:
        transform = np.asarray(self.transform_wc, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("transform_wc must have shape 4x4")
        object.__setattr__(self, "transform_wc", transform)


def write_tum_trajectory(path: str | Path, poses: Iterable[PoseStamped]) -> None:
    """Write poses in TUM RGB-D format: `timestamp tx ty tz qx qy qz qw`."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pose in poses:
        translation = pose.transform_wc[:3, 3]
        quaternion = Rotation.from_matrix(pose.transform_wc[:3, :3]).as_quat()
        values = [pose.timestamp, *translation, *quaternion]
        lines.append(" ".join(_format_float(value) for value in values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_tum_trajectory(path: str | Path) -> list[PoseStamped]:
    """Read TUM RGB-D trajectory format into `T_wc` poses."""

    poses: list[PoseStamped] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 8:
            raise ValueError(f"line {line_number}: expected 8 columns, got {len(parts)}")
        timestamp, tx, ty, tz, qx, qy, qz, qw = (float(value) for value in parts)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        transform[:3, 3] = [tx, ty, tz]
        poses.append(PoseStamped(timestamp=timestamp, transform_wc=transform))
    return poses


def write_kitti_trajectory(path: str | Path, poses: Iterable[np.ndarray | PoseStamped]) -> None:
    """Write poses in KITTI odometry format: one flattened `3x4` `T_wc` per line."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pose in poses:
        transform = pose.transform_wc if isinstance(pose, PoseStamped) else pose
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("KITTI trajectory poses must have shape 4x4")
        values = transform[:3, :4].reshape(-1)
        lines.append(" ".join(_format_float(value) for value in values))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _format_float(value: float) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".")
