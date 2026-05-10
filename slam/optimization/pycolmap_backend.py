"""Optional PyCOLMAP reference backend entry points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slam.optimization.gtsam_backend import OptionalBackendDependencyError


@dataclass(frozen=True)
class PyCOLMAPReconstructionResult:
    """Summary of a PyCOLMAP sparse reconstruction run."""

    image_dir: Path
    output_dir: Path
    database_path: Path
    reconstruction_ids: tuple[int, ...]
    reconstruction_paths: tuple[Path, ...]
    summaries: tuple[str, ...]
    reconstructions: tuple[object, ...]

    @property
    def reconstruction_count(self) -> int:
        return len(self.reconstructions)


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


def run_pycolmap_reconstruction(
    image_dir: str | Path,
    output_dir: str | Path,
    *,
    database_path: str | Path | None = None,
    image_names: Sequence[str] = (),
    camera_mode: object | None = None,
    reader_options: object | None = None,
    extraction_options: object | None = None,
    matching_method: str = "exhaustive",
    matching_options: object | None = None,
    pairing_options: object | None = None,
    verification_options: object | None = None,
    mapper_options: object | None = None,
    device: object | None = None,
    input_path: str | Path | None = None,
    write: bool = True,
) -> PyCOLMAPReconstructionResult:
    """Run PyCOLMAP's sparse reconstruction reference pipeline on an image directory."""

    pycolmap = require_pycolmap()
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = Path(database_path) if database_path is not None else output_dir / "database.db"

    extract_kwargs = _drop_none(
        image_names=tuple(image_names),
        camera_mode=camera_mode,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=device,
    )
    pycolmap.extract_features(database_path, image_dir, **extract_kwargs)
    _match_features(
        pycolmap,
        database_path,
        method=matching_method,
        matching_options=matching_options,
        pairing_options=pairing_options,
        verification_options=verification_options,
        device=device,
    )
    mapper_kwargs = _drop_none(options=mapper_options, input_path=input_path)
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_dir, **mapper_kwargs)

    entries = _reconstruction_entries(reconstructions)
    reconstruction_paths = _write_reconstructions(entries, output_dir) if write else tuple()
    return PyCOLMAPReconstructionResult(
        image_dir=image_dir,
        output_dir=output_dir,
        database_path=database_path,
        reconstruction_ids=tuple(reconstruction_id for reconstruction_id, _ in entries),
        reconstruction_paths=reconstruction_paths,
        summaries=tuple(_reconstruction_summary(reconstruction) for _, reconstruction in entries),
        reconstructions=tuple(reconstruction for _, reconstruction in entries),
    )


def _match_features(
    pycolmap,
    database_path: Path,
    *,
    method: str,
    matching_options: object | None,
    pairing_options: object | None,
    verification_options: object | None,
    device: object | None,
) -> None:
    methods = {
        "exhaustive": "match_exhaustive",
        "image_pairs": "match_image_pairs",
        "sequential": "match_sequential",
        "spatial": "match_spatial",
        "vocabtree": "match_vocabtree",
    }
    method = method.lower()
    if method not in methods:
        options = ", ".join(sorted(methods))
        raise ValueError(f"matching_method must be one of: {options}")
    match_fn = getattr(pycolmap, methods[method])
    match_fn(
        database_path,
        **_drop_none(
            matching_options=matching_options,
            pairing_options=pairing_options,
            verification_options=verification_options,
            device=device,
        ),
    )


def _drop_none(**kwargs: object) -> dict[str, object]:
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and not (isinstance(value, tuple) and len(value) == 0)
    }


def _reconstruction_entries(reconstructions: Mapping[int, object] | Sequence[object]) -> tuple[tuple[int, object], ...]:
    if isinstance(reconstructions, Mapping):
        return tuple(
            (int(reconstruction_id), reconstruction) for reconstruction_id, reconstruction in reconstructions.items()
        )
    return tuple((index, reconstruction) for index, reconstruction in enumerate(reconstructions))


def _write_reconstructions(entries: tuple[tuple[int, object], ...], output_dir: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for reconstruction_id, reconstruction in entries:
        path = output_dir if len(entries) == 1 else output_dir / str(reconstruction_id)
        path.mkdir(parents=True, exist_ok=True)
        reconstruction.write(path)
        paths.append(path)
    return tuple(paths)


def _reconstruction_summary(reconstruction: Any) -> str:
    if hasattr(reconstruction, "summary"):
        return str(reconstruction.summary())
    return str(reconstruction)
