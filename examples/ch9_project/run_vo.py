"""Run a minimal monocular feature-VO trajectory over an image directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.features.opencv_features import SUPPORTED_FEATURES, match_images
from slam.io.datasets import list_image_sequence
from slam.io.trajectory import PoseStamped, write_kitti_trajectory, write_tum_trajectory
from slam.vo.two_view import estimate_essential, recover_relative_pose
from slam.vo.visual_odometry import VisualOdometryConfig, chain_relative_pose


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Directory containing an ordered image sequence.")
    parser.add_argument("--pattern", default="*.png", help="Glob pattern under --images.")
    parser.add_argument("--intrinsics", required=True, type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--matcher", choices=sorted(SUPPORTED_FEATURES), help="Override config matcher.")
    parser.add_argument("--translation-scale", type=float, default=1.0, help="Scale applied to unit recoverPose translations.")
    parser.add_argument("--output-tum", type=Path, help="Optional TUM trajectory output path.")
    parser.add_argument("--output-kitti", type=Path, help="Optional KITTI trajectory output path.")
    return parser.parse_args()


def _read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def main() -> None:
    args = _parse_args()
    config = VisualOdometryConfig.from_yaml(args.config)
    matcher = args.matcher or config.matcher
    fx, fy, cx, cy = args.intrinsics
    camera_matrix = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy).matrix
    frames = list_image_sequence(args.images, pattern=args.pattern)
    if len(frames) < 2:
        raise SystemExit("Need at least two images for VO.")

    pose_wc = np.eye(4, dtype=np.float64)
    trajectory = [PoseStamped(timestamp=float(frames[0].index), transform_wc=pose_wc)]
    successful_steps = 0

    for previous, current in zip(frames, frames[1:]):
        image0 = _read_grayscale(previous.image_path)
        image1 = _read_grayscale(current.image_path)
        matches = match_images(image0, image1, feature=matcher, max_features=config.max_features)
        if len(matches.matches) < config.min_matches:
            print(f"frame {current.index}: skipped, matches={len(matches.matches)}")
            trajectory.append(PoseStamped(timestamp=float(current.index), transform_wc=pose_wc.copy()))
            continue

        essential = estimate_essential(matches.points0, matches.points1, camera_matrix)
        pose = recover_relative_pose(essential.matrix, matches.points0, matches.points1, camera_matrix, mask=essential.mask)
        pose_wc = chain_relative_pose(
            pose_wc,
            pose.rotation_10,
            pose.translation_10,
            translation_scale=args.translation_scale,
        )
        successful_steps += 1
        trajectory.append(PoseStamped(timestamp=float(current.index), transform_wc=pose_wc.copy()))
        print(
            f"frame {current.index}: matches={len(matches.matches)} "
            f"essential_inliers={essential.inlier_count} pose_inliers={pose.inlier_count}"
        )

    if args.output_tum is not None:
        write_tum_trajectory(args.output_tum, trajectory)
        print(f"wrote TUM trajectory: {args.output_tum}")
    if args.output_kitti is not None:
        write_kitti_trajectory(args.output_kitti, trajectory)
        print(f"wrote KITTI trajectory: {args.output_kitti}")

    print(f"image count: {len(frames)}")
    print(f"successful relative pose steps: {successful_steps}")
    print("monocular scale note: translations use --translation-scale and are not metric unless scale is known")


if __name__ == "__main__":
    main()
