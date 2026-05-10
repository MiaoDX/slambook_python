"""Run smoke validation against upstream slambook sample data.

The runner expects a sparse or full checkout of gaoxiang12/slambook with the
sample paths used by this port. It writes a machine-readable JSON report while
also streaming a short per-case summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_NUMBER_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$", re.IGNORECASE)
_SHAPE_RE = re.compile(r"^\((\d+(?:,\s*\d+)*)\)$")


@dataclass(frozen=True)
class ValidationCase:
    """One external command used to validate a sample-data workflow."""

    name: str
    command: list[str]
    required_paths: tuple[Path, ...]


@dataclass(frozen=True)
class ValidationResult:
    """Serializable validation result for a single case."""

    name: str
    status: str
    command: list[str]
    returncode: int | None
    duration_seconds: float
    metrics: dict[str, Any]
    missing_paths: list[str]
    stdout_tail: str
    stderr_tail: str


def parse_stdout_metrics(stdout: str) -> dict[str, Any]:
    """Extract simple ``key: value`` and ``key=value`` metrics from CLI output."""

    metrics: dict[str, Any] = {}
    in_candidates = False
    candidate_count = 0

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_candidates:
            if line.startswith("index="):
                candidate_count += 1
                continue
            in_candidates = False
        if line == "candidates:":
            in_candidates = True
            continue

        if ":" in line:
            key_text, value_text = line.split(":", 1)
            line_key = _normalize_key(key_text)
            if line_key and value_text.strip():
                metrics[line_key] = _coerce_metric_value(value_text.strip())
        else:
            line_key = ""

        for key_text, value_text in re.findall(r"([A-Za-z_][A-Za-z0-9_-]*)=([^\s]+)", line):
            key = _normalize_key(key_text)
            if line_key.startswith("frame_"):
                key = f"{line_key}_{key}"
            metrics[key] = _coerce_metric_value(value_text)

    if in_candidates or candidate_count:
        metrics["candidate_count"] = candidate_count
    return metrics


def build_validation_cases(
    *,
    upstream_root: Path,
    work_dir: Path,
    python: str,
    case_names: set[str],
) -> list[ValidationCase]:
    """Build validation cases for the requested upstream sample workflows."""

    ch7_dir = upstream_root / "ch7"
    ch12_data = upstream_root / "ch12" / "data"
    ch13_data = upstream_root / "ch13" / "dense_RGBD" / "data"
    vocabulary_path = work_dir / "ch12_vocabulary.npz"
    descriptor_path = work_dir / "ch12_descriptors.npy"
    cloud_path = work_dir / "ch13_cloud.ply"
    occupancy_path = work_dir / "ch13_occupancy.npz"

    cases = [
        ValidationCase(
            name="ch7_3d3d_pose",
            required_paths=(
                ch7_dir / "1.png",
                ch7_dir / "2.png",
                ch7_dir / "1_depth.png",
                ch7_dir / "2_depth.png",
            ),
            command=[
                python,
                "examples/ch7_feature_vo/pose_estimation_3d3d.py",
                "--image0",
                str(ch7_dir / "1.png"),
                "--image1",
                str(ch7_dir / "2.png"),
                "--depth0",
                str(ch7_dir / "1_depth.png"),
                "--depth1",
                str(ch7_dir / "2_depth.png"),
            ],
        ),
        ValidationCase(
            name="ch9_local_map_vo",
            required_paths=(
                ch7_dir / "1.png",
                ch7_dir / "2.png",
                ch7_dir / "1_depth.png",
                ch7_dir / "2_depth.png",
            ),
            command=[
                python,
                "examples/ch9_project/run_local_map_vo.py",
                "--images",
                str(ch7_dir),
                "--depths",
                str(ch7_dir),
                "--image-pattern",
                "[0-9].png",
                "--depth-pattern",
                "*_depth.png",
                "--intrinsics",
                "520.9",
                "521.0",
                "325.1",
                "249.7",
                "--max-new-points",
                "300",
            ],
        ),
        ValidationCase(
            name="ch12_train_vocabulary",
            required_paths=(ch12_data,),
            command=[
                python,
                "examples/ch12_loop_closure/train_vocabulary.py",
                "--images",
                str(ch12_data),
                "--pattern",
                "*.png",
                "--words",
                "8",
                "--max-iter",
                "20",
                "--output",
                str(vocabulary_path),
            ],
        ),
        ValidationCase(
            name="ch12_build_descriptors",
            required_paths=(ch12_data, vocabulary_path),
            command=[
                python,
                "examples/ch12_loop_closure/build_descriptors.py",
                "--images",
                str(ch12_data),
                "--pattern",
                "*.png",
                "--vocabulary",
                str(vocabulary_path),
                "--output",
                str(descriptor_path),
            ],
        ),
        ValidationCase(
            name="ch12_retrieve_candidates",
            required_paths=(descriptor_path,),
            command=[
                python,
                "examples/ch12_loop_closure/retrieve_candidates.py",
                "--descriptors",
                str(descriptor_path),
                "--current-index",
                "9",
                "--top-k",
                "3",
                "--temporal-window",
                "1",
            ],
        ),
        ValidationCase(
            name="ch13_rgbd_fusion",
            required_paths=(
                ch13_data / "color",
                ch13_data / "depth",
                ch13_data / "pose.txt",
            ),
            command=[
                python,
                "examples/ch13_dense_mapping/rgbd_fusion.py",
                "--color-dir",
                str(ch13_data / "color"),
                "--depth-dir",
                str(ch13_data / "depth"),
                "--depth-pattern",
                "*.pgm",
                "--pose-file",
                str(ch13_data / "pose.txt"),
                "--output",
                str(cloud_path),
                "--intrinsics",
                "518.0",
                "519.0",
                "325.5",
                "253.5",
                "--depth-scale",
                "1000",
                "--voxel-size",
                "0.02",
                "--occupancy-output",
                str(occupancy_path),
                "--occupancy-voxel-size",
                "0.05",
            ],
        ),
    ]
    if "all" in case_names:
        return cases
    return [case for case in cases if case.name in case_names or _case_group(case.name) in case_names]


def run_validation_case(case: ValidationCase, *, repo_root: Path, env: dict[str, str]) -> ValidationResult:
    """Run one validation case and return a serializable result."""

    missing_paths = [str(path) for path in case.required_paths if not path.exists()]
    if missing_paths:
        return ValidationResult(
            name=case.name,
            status="missing_input",
            command=case.command,
            returncode=None,
            duration_seconds=0.0,
            metrics={},
            missing_paths=missing_paths,
            stdout_tail="",
            stderr_tail="",
        )

    start = time.monotonic()
    completed = subprocess.run(
        case.command,
        cwd=repo_root,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    duration_seconds = time.monotonic() - start
    return ValidationResult(
        name=case.name,
        status="passed" if completed.returncode == 0 else "failed",
        command=case.command,
        returncode=completed.returncode,
        duration_seconds=round(duration_seconds, 3),
        metrics=parse_stdout_metrics(completed.stdout),
        missing_paths=[],
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
    )


def build_report(
    *,
    results: list[ValidationResult],
    upstream_root: Path,
    repo_root: Path,
    work_dir: Path,
) -> dict[str, Any]:
    """Build a JSON-serializable validation report."""

    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "upstream_root": str(upstream_root),
        "work_dir": str(work_dir),
        "summary": {
            "case_count": len(results),
            "status_counts": status_counts,
            "passed": all(result.status == "passed" for result in results),
        },
        "results": [asdict(result) for result in results],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/slambook-python-validation"))
    parser.add_argument("--report", type=Path, help="Output JSON report path. Defaults inside --work-dir.")
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "all",
            "ch7",
            "ch9",
            "ch12",
            "ch13",
            "ch7_3d3d_pose",
            "ch9_local_map_vo",
            "ch12_train_vocabulary",
            "ch12_build_descriptors",
            "ch12_retrieve_candidates",
            "ch13_rgbd_fusion",
        ),
        default=None,
        help="Run one case or chapter group. May be repeated.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child commands.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    upstream_root = args.upstream_root.resolve()
    work_dir = args.work_dir.resolve()
    report_path = args.report.resolve() if args.report is not None else work_dir / "upstream_validation_report.json"
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cases = build_validation_cases(
        upstream_root=upstream_root,
        work_dir=work_dir,
        python=args.python,
        case_names=set(args.case or ["all"]),
    )
    if not cases:
        raise SystemExit("No validation cases selected.")

    results = []
    for case in cases:
        result = run_validation_case(case, repo_root=repo_root, env=env)
        results.append(result)
        print(f"{result.name}: {result.status}")
        if result.metrics:
            print(f"  metrics: {json.dumps(result.metrics, sort_keys=True)}")
        if result.missing_paths:
            print(f"  missing paths: {', '.join(result.missing_paths)}")

    report = build_report(results=results, upstream_root=upstream_root, repo_root=repo_root, work_dir=work_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote validation report: {report_path}")

    if not report["summary"]["passed"]:
        raise SystemExit(1)


def _case_group(case_name: str) -> str:
    return case_name.split("_", 1)[0]


def _coerce_metric_value(value: str) -> Any:
    value = value.strip()
    shape_match = _SHAPE_RE.match(value)
    if shape_match is not None:
        return [int(part.strip()) for part in shape_match.group(1).split(",")]
    if _NUMBER_RE.match(value):
        number = float(value)
        if number.is_integer() and "." not in value and "e" not in value.lower():
            return int(number)
        return number
    return value


def _normalize_key(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("3d-3d", "3d_3d")
    text = text.replace("rmse m", "rmse_m")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _tail(text: str, *, max_lines: int = 20) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


if __name__ == "__main__":
    main()
