"""Shared helpers for Chapter 7 command-line examples."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera.pinhole import CameraIntrinsics


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="Camera intrinsics as fx fy cx cy.",
    )
    parser.add_argument(
        "--camera-matrix",
        type=float,
        nargs=9,
        metavar=("K00", "K01", "K02", "K10", "K11", "K12", "K20", "K21", "K22"),
        help="Camera matrix as 9 row-major values. Overrides --intrinsics.",
    )


def read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def camera_matrix_from_args(args: argparse.Namespace, image_shape: tuple[int, int]) -> np.ndarray:
    if args.camera_matrix is not None:
        return np.asarray(args.camera_matrix, dtype=np.float64).reshape(3, 3)

    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
    else:
        height, width = image_shape
        fx = fy = float(max(width, height))
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy).matrix


def format_matrix(matrix: np.ndarray) -> str:
    return np.array2string(np.asarray(matrix), precision=6, suppress_small=True)


def object_points_from_depth(
    pixels0: np.ndarray,
    pixels1: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project first-image pixels with valid depth and keep matching pixels."""

    pixels0 = np.asarray(pixels0, dtype=np.float64)
    pixels1 = np.asarray(pixels1, dtype=np.float64)
    if pixels0.shape != pixels1.shape:
        raise ValueError("pixels0 and pixels1 must have the same shape")
    if pixels0.ndim != 2 or pixels0.shape[1] != 2:
        raise ValueError("pixels0 and pixels1 must be Nx2 arrays")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")

    rounded = np.rint(pixels0).astype(np.int64)
    height, width = depth.shape[:2]
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )

    raw_depth = np.zeros(len(pixels0), dtype=np.float64)
    valid_indices = np.flatnonzero(in_bounds)
    raw_depth[valid_indices] = depth[rounded[valid_indices, 1], rounded[valid_indices, 0]].astype(np.float64)
    metric_depth = raw_depth / depth_scale
    valid = in_bounds & np.isfinite(metric_depth) & (metric_depth > 0.0)

    points_3d = intrinsics.pixel_to_camera(pixels0[valid], metric_depth[valid])
    return points_3d, pixels1[valid], valid
