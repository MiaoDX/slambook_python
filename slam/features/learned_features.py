"""Optional learned feature matcher adapters."""

from __future__ import annotations

import numpy as np


class OptionalFeatureDependencyError(ImportError):
    """Raised when an optional learned matching dependency is unavailable."""


def _missing_dependency(name: str, extra: str = "modern") -> OptionalFeatureDependencyError:
    return OptionalFeatureDependencyError(
        f"{name} is an optional matcher backend. Install it with `pip install -e .[{extra}]` "
        "and make sure the package supports your Python/platform."
    )


class LightGlueMatcher:
    """Adapter placeholder for LightGlue-based matching.

    The class validates that optional dependencies are present at construction
    time. A full tensor adapter can be added without changing the public matcher
    selection surface used by examples.
    """

    name = "lightglue"

    def __init__(self) -> None:
        try:
            import lightglue  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise _missing_dependency("LightGlue") from exc

    def match_images(self, image0: np.ndarray, image1: np.ndarray):
        raise NotImplementedError("LightGlue tensor matching adapter is not implemented yet.")


class LoFTRMatcher:
    """Adapter placeholder for Kornia LoFTR matching."""

    name = "loftr"

    def __init__(self) -> None:
        try:
            import kornia  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            raise _missing_dependency("Kornia LoFTR") from exc

    def match_images(self, image0: np.ndarray, image1: np.ndarray):
        raise NotImplementedError("LoFTR tensor matching adapter is not implemented yet.")


def create_learned_matcher(name: str):
    """Create an optional learned matcher by name."""

    normalized = name.lower()
    if normalized == "lightglue":
        return LightGlueMatcher()
    if normalized == "loftr":
        return LoFTRMatcher()
    raise ValueError("learned matcher must be 'lightglue' or 'loftr'")
