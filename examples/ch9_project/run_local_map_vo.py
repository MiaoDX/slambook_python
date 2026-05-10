"""Run depth-assisted local-map visual odometry over an RGB-D image sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.features.opencv_features import SUPPORTED_FEATURES
from slam.io.datasets import list_image_sequence
from slam.io.trajectory import PoseStamped, write_kitti_trajectory, write_tum_trajectory
from slam.vo.visual_odometry import Camera, VisualOdometry, VisualOdometryConfig, insert_depth_map_points


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Directory containing an ordered image sequence.")
    parser.add_argument("--depths", required=True, type=Path, help="Directory containing aligned depth images.")
    parser.add_argument("--image-pattern", default="*.png")
    parser.add_argument("--depth-pattern", default="*.png")
    parser.add_argument("--intrinsics", required=True, type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--matcher", choices=sorted(SUPPORTED_FEATURES), help="Override config matcher.")
    parser.add_argument("--depth-scale", type=float, default=5000.0)
    parser.add_argument("--max-new-points", type=int, default=500)
    parser.add_argument("--output-tum", type=Path, help="Optional TUM trajectory output path.")
    parser.add_argument("--output-kitti", type=Path, help="Optional KITTI trajectory output path.")
    return parser.parse_args()


def _read_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Could not read image: {path}")
    return image


def _read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def main() -> None:
    args = _parse_args()
    config = VisualOdometryConfig.from_yaml(args.config)
    if args.matcher is not None:
        config = VisualOdometryConfig.from_mapping({**config.__dict__, "matcher": args.matcher})
    fx, fy, cx, cy = args.intrinsics
    camera = Camera(CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy))
    vo = VisualOdometry(camera=camera, config=config)
    image_frames = list_image_sequence(args.images, pattern=args.image_pattern)
    depth_frames = list_image_sequence(args.depths, pattern=args.depth_pattern)
    if len(image_frames) != len(depth_frames):
        raise SystemExit(f"Image/depth frame count mismatch: {len(image_frames)} vs {len(depth_frames)}.")
    if not image_frames:
        raise SystemExit("Need at least one RGB-D frame.")

    first = vo.create_frame(0, float(image_frames[0].index), _read_grayscale(image_frames[0].image_path))
    vo.insert_keyframe(first)
    inserted = insert_depth_map_points(
        first,
        vo.map,
        camera,
        _read_depth(depth_frames[0].image_path),
        depth_scale=args.depth_scale,
        max_points=args.max_new_points,
    )
    trajectory = [PoseStamped(timestamp=float(image_frames[0].index), transform_wc=first.pose_wc.copy())]
    print(f"frame 0: keyframe=1 inserted_points={inserted} map_points={len(vo.map.points)}")

    for image_frame, depth_frame in zip(image_frames[1:], depth_frames[1:]):
        frame = vo.create_frame(
            image_frame.index,
            float(image_frame.index),
            _read_grayscale(image_frame.image_path),
        )
        result = vo.track_local_map(frame)
        if not result.success:
            if vo.current_frame is not None:
                frame.set_pose(vo.current_frame.pose_wc.copy())
            print(f"frame {frame.id}: tracking_failed matches={len(result.matches)} message={result.message}")
        elif frame.id in vo.map.keyframes:
            inserted = insert_depth_map_points(
                frame,
                vo.map,
                camera,
                _read_depth(depth_frame.image_path),
                depth_scale=args.depth_scale,
                max_points=args.max_new_points,
            )
            print(
                f"frame {frame.id}: tracked inliers={result.pnp.inlier_count if result.pnp else 0} "
                f"keyframe=1 inserted_points={inserted} map_points={len(vo.map.points)}"
            )
        else:
            print(f"frame {frame.id}: tracked inliers={result.pnp.inlier_count if result.pnp else 0} keyframe=0")
        trajectory.append(PoseStamped(timestamp=float(image_frame.index), transform_wc=frame.pose_wc.copy()))

    if args.output_tum is not None:
        write_tum_trajectory(args.output_tum, trajectory)
        print(f"wrote TUM trajectory: {args.output_tum}")
    if args.output_kitti is not None:
        write_kitti_trajectory(args.output_kitti, trajectory)
        print(f"wrote KITTI trajectory: {args.output_kitti}")
    print(f"image count: {len(image_frames)}")
    print(f"keyframe count: {len(vo.map.keyframes)}")
    print(f"map point count: {len(vo.map.points)}")


if __name__ == "__main__":
    main()
