"""Build simple OpenCV global descriptors for an image sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.features.opencv_features import SUPPORTED_FEATURES
from slam.io.datasets import list_image_sequence
from slam.io.image_retrieval import opencv_global_descriptor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature", choices=sorted(SUPPORTED_FEATURES), default="orb")
    parser.add_argument("--max-features", type=int, default=1000)
    return parser.parse_args()


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def main() -> None:
    args = _parse_args()
    frames = list_image_sequence(args.images, pattern=args.pattern)
    if not frames:
        raise SystemExit("No images matched the requested pattern.")

    descriptors = []
    for frame in frames:
        descriptors.append(
            opencv_global_descriptor(
                _read_gray(frame.image_path),
                feature=args.feature,
                max_features=args.max_features,
            )
        )
    matrix = np.vstack(descriptors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix)

    print(f"images: {args.images}")
    print(f"pattern: {args.pattern}")
    print(f"feature: {args.feature}")
    print(f"descriptor shape: {matrix.shape}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
