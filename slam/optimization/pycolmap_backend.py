"""Optional PyCOLMAP reference backend entry points."""

from __future__ import annotations

from slam.optimization.gtsam_backend import OptionalBackendDependencyError


def require_pycolmap():
    """Import PyCOLMAP or raise with project-specific install guidance."""

    try:
        import pycolmap  # type: ignore
    except ImportError as exc:
        raise OptionalBackendDependencyError(
            "PyCOLMAP is an optional reference backend. Install it with `pip install -e .[modern]` "
            "and verify wheel support for your Python/platform."
        ) from exc
    return pycolmap


def run_pycolmap_reconstruction(*args, **kwargs):
    """Placeholder for a future PyCOLMAP reconstruction reference adapter."""

    require_pycolmap()
    raise NotImplementedError("PyCOLMAP reconstruction adapter is not implemented yet.")
