"""Check availability for an optional PyCOLMAP reconstruction reference path."""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", help="Image directory for a future PyCOLMAP reconstruction run.")
    return parser.parse_args()


def main() -> None:
    _parse_args()
    try:
        import pycolmap  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyCOLMAP is an optional reference backend. Install it with `pip install -e .[modern]` "
            "and verify wheel support for your Python/platform."
        ) from exc
    raise SystemExit("PyCOLMAP is installed; reconstruction adapter implementation is pending.")


if __name__ == "__main__":
    main()
