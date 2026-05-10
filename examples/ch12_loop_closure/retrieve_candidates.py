"""Retrieve loop-closure candidates from a descriptor matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from slam.io.image_retrieval import retrieve_loop_candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptors", required=True, type=Path, help="Path to an NxD .npy descriptor matrix.")
    parser.add_argument("--current-index", required=True, type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--temporal-window", type=int, default=10)
    parser.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    descriptors = np.load(args.descriptors)
    candidates = retrieve_loop_candidates(
        descriptors,
        current_index=args.current_index,
        top_k=args.top_k,
        temporal_window=args.temporal_window,
        metric=args.metric,
    )

    print(f"descriptors: {args.descriptors}")
    print(f"descriptor shape: {descriptors.shape}")
    print(f"current index: {args.current_index}")
    print(f"temporal window: {args.temporal_window}")
    print(f"metric: {args.metric}")
    print("candidates:")
    for candidate in candidates:
        print(f"  index={candidate.index} score={candidate.score:.9f}")


if __name__ == "__main__":
    main()
