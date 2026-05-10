"""Run the optional PyCOLMAP sparse reconstruction reference pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from slam.optimization.gtsam_backend import OptionalBackendDependencyError
from slam.optimization.pycolmap_backend import run_pycolmap_reconstruction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Directory containing input images.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for database and sparse model.")
    parser.add_argument("--database", type=Path, help="Optional COLMAP database path. Defaults to OUTPUT/database.db.")
    parser.add_argument(
        "--matching-method",
        choices=["exhaustive", "image_pairs", "sequential", "spatial", "vocabtree"],
        default="exhaustive",
        help="PyCOLMAP matcher to run after feature extraction.",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not call Reconstruction.write on returned models.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        result = run_pycolmap_reconstruction(
            args.images,
            args.output,
            database_path=args.database,
            matching_method=args.matching_method,
            write=not args.no_write,
        )
    except OptionalBackendDependencyError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"images: {result.image_dir}")
    print(f"database: {result.database_path}")
    print(f"reconstructions: {result.reconstruction_count}")
    for reconstruction_id, summary in zip(result.reconstruction_ids, result.summaries):
        print(f"[{reconstruction_id}] {summary}")


if __name__ == "__main__":
    main()
