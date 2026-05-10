"""Matplotlib visualization helpers."""

from __future__ import annotations

from pathlib import Path

from slam.viz.open3d_viz import OptionalVisualizationDependencyError


def require_matplotlib():
    """Import Matplotlib pyplot or raise with install guidance."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise OptionalVisualizationDependencyError(
            "Matplotlib is optional. Install it with `pip install -e .[core]` "
            "and verify wheel support for your Python/platform."
        ) from exc
    return plt


def trajectory_xyz(poses: list[np.ndarray]) -> np.ndarray:
    """Extract `Nx3` translation positions from `T_wc` poses."""

    import numpy as np

    positions = []
    for pose in poses:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError("trajectory poses must have shape 4x4")
        positions.append(pose[:3, 3])
    return np.asarray(positions, dtype=np.float64).reshape(-1, 3)


def plot_trajectory(poses: list[np.ndarray], *, ax=None, label: str | None = None):
    """Plot a 3D trajectory with Matplotlib and return the axes."""

    xyz = trajectory_xyz(poses)
    if ax is None:
        plt = require_matplotlib()
        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")
    if len(xyz):
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if label is not None:
        ax.legend()
    return ax


def save_trajectory_plot(path: str | Path, poses: list[np.ndarray], *, label: str | None = None) -> None:
    """Save a 3D trajectory plot to an image file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = require_matplotlib()
    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    plot_trajectory(poses, ax=ax, label=label)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
