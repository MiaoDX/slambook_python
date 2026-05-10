"""Compare BAL camera observations against PyCOLMAP absolute-pose estimates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from slam.optimization.bundle_adjustment import read_bal_problem
from slam.optimization.gtsam_backend import OptionalBackendDependencyError
from slam.vo.pycolmap_backend import estimate_absolute_pose_pycolmap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bal", required=True, type=Path, help="Path to a BAL text problem.")
    parser.add_argument("--max-cameras", type=int, help="Maximum number of BAL cameras to check.")
    parser.add_argument("--image-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=(1, 1))
    parser.add_argument("--principal-point", nargs=2, type=float, metavar=("CX", "CY"), default=(0.0, 0.0))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    problem = read_bal_problem(args.bal)
    print(f"BAL file: {args.bal}")
    print(f"camera count: {len(problem.camera_params)}")
    print(f"point count: {len(problem.points_3d)}")
    print(f"observation count: {len(problem.observations)}")

    observations_by_camera: dict[int, list] = {}
    for observation in problem.observations:
        observations_by_camera.setdefault(observation.camera_index, []).append(observation)

    checked = 0
    for camera_index, observations in sorted(observations_by_camera.items()):
        if args.max_cameras is not None and checked >= args.max_cameras:
            break
        if len(observations) < 4:
            print(f"camera {camera_index}: skipped, need at least 4 observations")
            continue

        camera_params = problem.camera_params[camera_index]
        camera_matrix = _camera_matrix_from_bal(camera_params, principal_point=args.principal_point)
        points_3d = np.asarray([problem.points_3d[obs.point_index] for obs in observations], dtype=np.float64)
        points_2d = np.asarray([obs.xy + np.asarray(args.principal_point) for obs in observations], dtype=np.float64)
        try:
            result = estimate_absolute_pose_pycolmap(
                points_3d,
                points_2d,
                camera_matrix,
                image_size=tuple(args.image_size),
            )
        except OptionalBackendDependencyError as exc:
            raise SystemExit(str(exc)) from exc
        except RuntimeError as exc:
            print(f"camera {camera_index}: PyCOLMAP failed: {exc}")
            checked += 1
            continue

        print(
            f"camera {camera_index}: observations={len(observations)} "
            f"inliers={result.inlier_count} translation={result.translation.reshape(3).tolist()}"
        )
        checked += 1

    if checked == 0:
        print("No BAL cameras had enough observations for PyCOLMAP absolute-pose estimation.")


def _camera_matrix_from_bal(camera_params: np.ndarray, *, principal_point: tuple[float, float]) -> np.ndarray:
    focal = float(np.asarray(camera_params, dtype=np.float64).reshape(9)[6])
    cx, cy = principal_point
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


if __name__ == "__main__":
    main()
