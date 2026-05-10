"""Optional GTSAM backend entry points."""

from __future__ import annotations


class OptionalBackendDependencyError(ImportError):
    """Raised when an optional optimization backend is unavailable."""


def require_gtsam():
    """Import GTSAM or raise with project-specific install guidance."""

    try:
        import gtsam  # type: ignore
    except ImportError as exc:
        raise OptionalBackendDependencyError(
            "GTSAM is an optional backend. Install it with `pip install -e .[backend]` "
            "and verify that GTSAM wheels are available for your Python/platform."
        ) from exc
    return gtsam


def optimize_pose_graph_gtsam(*args, **kwargs):
    """Placeholder for a future GTSAM pose graph optimizer."""

    require_gtsam()
    raise NotImplementedError("GTSAM pose graph optimization adapter is not implemented yet.")


def optimize_bundle_adjustment_gtsam(*args, **kwargs):
    """Placeholder for a future GTSAM bundle adjustment optimizer."""

    require_gtsam()
    raise NotImplementedError("GTSAM bundle adjustment adapter is not implemented yet.")
