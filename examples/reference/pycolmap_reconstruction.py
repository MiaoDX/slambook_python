"""Check availability for an optional PyCOLMAP reconstruction reference path."""

from __future__ import annotations

import argparse

from slam.optimization.gtsam_backend import OptionalBackendDependencyError
from slam.optimization.pycolmap_backend import require_pycolmap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", help="Image directory for a future PyCOLMAP reconstruction run.")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    try:
        require_pycolmap()
    except OptionalBackendDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    raise SystemExit("PyCOLMAP is installed; reconstruction adapter implementation is pending.")


if __name__ == "__main__":
    main()
