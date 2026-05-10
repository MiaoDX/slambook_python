"""Compare available matcher backends on one image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from ch7_common import read_grayscale
from slam.features.learned_features import OptionalFeatureDependencyError, create_learned_matcher
from slam.features.opencv_features import SUPPORTED_FEATURES, match_images
from slam.viz import OptionalVisualizationDependencyError, log_matches_rerun, require_rerun


MATCHERS = sorted(SUPPORTED_FEATURES | {"lightglue", "loftr"})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", required=True, type=Path)
    parser.add_argument("--image1", required=True, type=Path)
    parser.add_argument("--matchers", nargs="+", choices=MATCHERS, default=["orb", "sift"])
    parser.add_argument("--max-features", type=int, default=1000)
    parser.add_argument("--rerun", action="store_true", help="Log OpenCV match points and lines to Rerun.")
    parser.add_argument("--rerun-entity", default="world/matches", help="Rerun entity prefix for --rerun.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    image0 = read_grayscale(args.image0)
    image1 = read_grayscale(args.image1)

    print(f"image0: {args.image0}")
    print(f"image1: {args.image1}")
    if args.rerun:
        try:
            rr = require_rerun()
            rr.init("slambook_compare_matchers", spawn=True)
        except OptionalVisualizationDependencyError as exc:
            raise SystemExit(str(exc)) from exc
    for matcher in args.matchers:
        if matcher in SUPPORTED_FEATURES:
            result = match_images(image0, image1, feature=matcher, max_features=args.max_features)
            print(
                f"{matcher}: keypoints0={len(result.keypoints0)} "
                f"keypoints1={len(result.keypoints1)} matches={len(result.matches)}"
            )
            if args.rerun:
                log_matches_rerun(
                    f"{args.rerun_entity}/{matcher}",
                    result.points0,
                    result.points1,
                    color=[0, 128, 255],
                    radius=2.0,
                )
            continue

        try:
            learned_matcher = create_learned_matcher(matcher, max_features=args.max_features)
        except OptionalFeatureDependencyError as exc:
            print(f"{matcher}: unavailable: {exc}")
            continue

        result = learned_matcher.match_images(image0, image1)
        print(
            f"{matcher}: keypoints0={len(result.keypoints0)} "
            f"keypoints1={len(result.keypoints1)} matches={len(result.matches)}"
        )
        if args.rerun:
            log_matches_rerun(
                f"{args.rerun_entity}/{matcher}",
                result.points0,
                result.points1,
                color=[255, 128, 0],
                radius=2.0,
            )


if __name__ == "__main__":
    main()
