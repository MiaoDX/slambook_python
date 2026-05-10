"""Compare available matcher backends on one image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from ch7_common import read_grayscale
from slam.features.learned_features import OptionalFeatureDependencyError, create_learned_matcher
from slam.features.opencv_features import SUPPORTED_FEATURES, match_images


MATCHERS = sorted(SUPPORTED_FEATURES | {"lightglue", "loftr"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path)
    parser.add_argument("--image1", required=True, type=Path)
    parser.add_argument("--matchers", nargs="+", choices=MATCHERS, default=["orb", "sift"])
    parser.add_argument("--max-features", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image0 = read_grayscale(args.image0)
    image1 = read_grayscale(args.image1)

    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    for matcher in args.matchers:
        if matcher in SUPPORTED_FEATURES:
            result = match_images(image0, image1, feature=matcher, max_features=args.max_features)
            print(
                f"{matcher}: keypoints0={len(result.keypoints0)} "
                f"keypoints1={len(result.keypoints1)} matches={len(result.matches)}"
            )
            continue

        try:
            create_learned_matcher(matcher)
        except OptionalFeatureDependencyError as exc:
            print(f"{matcher}: unavailable: {exc}")
        else:
            print(f"{matcher}: dependency available, adapter implementation pending")


if __name__ == "__main__":
    main()
