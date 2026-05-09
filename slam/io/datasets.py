"""Dataset discovery helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageSequenceFrame:
    """One image in a sorted image sequence."""

    index: int
    image_path: Path


@dataclass(frozen=True)
class TumRgbdFrame:
    """Associated TUM RGB-D color/depth frame."""

    timestamp: float
    rgb_path: Path
    depth_path: Path
    depth_timestamp: float


def list_image_sequence(directory: str | Path, *, pattern: str = "*.png") -> list[ImageSequenceFrame]:
    """List image files sorted lexicographically under `directory`."""

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    paths = sorted(path for path in directory.glob(pattern) if path.is_file())
    return [ImageSequenceFrame(index=index, image_path=path) for index, path in enumerate(paths)]


def associate_tum_rgbd(
    rgb_txt: str | Path,
    depth_txt: str | Path,
    *,
    max_difference: float = 0.02,
) -> list[TumRgbdFrame]:
    """Associate TUM RGB-D `rgb.txt` and `depth.txt` files by nearest timestamp."""

    rgb_txt = Path(rgb_txt)
    depth_txt = Path(depth_txt)
    rgb_entries = _read_tum_index(rgb_txt)
    depth_entries = _read_tum_index(depth_txt)
    depth_unused = set(range(len(depth_entries)))
    frames: list[TumRgbdFrame] = []

    for rgb_time, rgb_relpath in rgb_entries:
        best_index = None
        best_difference = None
        for depth_index in list(depth_unused):
            depth_time, _ = depth_entries[depth_index]
            difference = abs(rgb_time - depth_time)
            if best_difference is None or difference < best_difference:
                best_index = depth_index
                best_difference = difference

        if best_index is None or best_difference is None or best_difference > max_difference:
            continue

        depth_unused.remove(best_index)
        depth_time, depth_relpath = depth_entries[best_index]
        frames.append(
            TumRgbdFrame(
                timestamp=rgb_time,
                rgb_path=(rgb_txt.parent / rgb_relpath).resolve(),
                depth_path=(depth_txt.parent / depth_relpath).resolve(),
                depth_timestamp=depth_time,
            )
        )

    return frames


def _read_tum_index(path: Path) -> list[tuple[float, Path]]:
    entries: list[tuple[float, Path]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            raise ValueError(f"{path}:{line_number}: expected at least 2 columns")
        entries.append((float(parts[0]), Path(parts[1])))
    return entries
