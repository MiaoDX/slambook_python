"""Train a small BoW visual vocabulary from OpenCV local descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.features.opencv_features import SUPPORTED_FEATURES, detect_and_compute
from slam.io.datasets import list_image_sequence
from slam.io.image_retrieval import train_visual_vocabulary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature", choices=sorted(SUPPORTED_FEATURES), default="orb")
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--words", type=int, default=64, help="Number of visual words.")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
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

    descriptor_blocks = []
    for frame in frames:
        _, descriptors = detect_and_compute(
            _read_gray(frame.image_path),
            feature=args.feature,
            max_features=args.max_features,
        )
        if descriptors is not None and len(descriptors):
            descriptor_blocks.append(descriptors)
    if not descriptor_blocks:
        raise SystemExit("No local descriptors were found for vocabulary training.")
    descriptors = np.vstack(descriptor_blocks)
    vocabulary = train_visual_vocabulary(
        descriptors,
        word_count=args.words,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    vocabulary.save(args.output)

    print(f"images: {args.images}")
    print(f"pattern: {args.pattern}")
    print(f"feature: {args.feature}")
    print(f"image count: {len(frames)}")
    print(f"local descriptor count: {len(descriptors)}")
    print(f"visual words: {vocabulary.word_count}")
    print(f"descriptor dimension: {vocabulary.descriptor_dim}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
