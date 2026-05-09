"""Inspect and optionally write basic OpenCV image conversions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path, help="Path to the input image.")
    parser.add_argument("--output-gray", type=Path, help="Optional path for a grayscale copy.")
    parser.add_argument("--output-edges", type=Path, help="Optional path for Canny edges.")
    parser.add_argument("--canny-thresholds", type=float, nargs=2, default=(80.0, 160.0), metavar=("LOW", "HIGH"))
    return parser.parse_args()


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise SystemExit(f"Could not write image: {path}")


def main() -> None:
    args = _parse_args()
    bgr = _read_image(args.image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    edges = cv2.Canny(gray, args.canny_thresholds[0], args.canny_thresholds[1])

    print(f"image: {args.image}")
    print(f"shape HxWxC: {bgr.shape}")
    print(f"dtype: {bgr.dtype}")
    print(f"BGR min/max: {int(bgr.min())} {int(bgr.max())}")
    print(f"gray shape: {gray.shape}")
    print(f"gray min/mean/max: {int(gray.min())} {float(gray.mean()):.3f} {int(gray.max())}")
    print(f"RGB first pixel: {rgb[0, 0].tolist()}")
    print(f"edge pixel count: {int(np.count_nonzero(edges))}")

    if args.output_gray is not None:
        _write_image(args.output_gray, gray)
        print(f"wrote grayscale: {args.output_gray}")
    if args.output_edges is not None:
        _write_image(args.output_edges, edges)
        print(f"wrote edges: {args.output_edges}")


if __name__ == "__main__":
    main()
