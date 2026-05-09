"""Point cloud file helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_ply_ascii(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write an ASCII PLY point cloud.

    `points` must be `Nx3`. `colors`, when provided, must be `Nx3` RGB values
    in either `uint8` or numeric range compatible with clipping to `[0, 255]`.
    """

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")

    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) != len(points):
            raise ValueError("colors must be an Nx3 array with one color per point")
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if colors is not None:
        lines.extend(
            [
                "property uchar red",
                "property uchar green",
                "property uchar blue",
            ]
        )
    lines.append("end_header")

    for index, point in enumerate(points):
        if colors is None:
            lines.append(f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f}")
        else:
            color = colors[index]
            lines.append(
                f"{point[0]:.9f} {point[1]:.9f} {point[2]:.9f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
