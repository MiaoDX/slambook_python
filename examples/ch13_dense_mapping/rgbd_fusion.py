"""Fuse one or more RGB-D frames with known poses and write PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.io.datasets import list_image_sequence
from slam.io.trajectory import read_slambook_pose_file
from slam.mapping.pointcloud import (
    estimate_normals,
    fuse_point_clouds,
    occupancy_voxel_grid,
    voxel_downsample,
    write_occupancy_npz,
    write_ply_ascii,
)
from slam.mapping.rgbd import rgbd_to_point_cloud
from slam.viz import (
    OptionalVisualizationDependencyError,
    log_points_rerun,
    reconstruct_mesh_poisson,
    require_rerun,
    write_triangle_mesh_open3d,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    single = parser.add_argument_group("single frame")
    single.add_argument("--color", type=Path, help="Path to an RGB/BGR color image.")
    single.add_argument("--depth", type=Path, help="Path to a depth image aligned to color.")
    sequence = parser.add_argument_group("known-pose sequence")
    sequence.add_argument("--color-dir", type=Path, help="Directory of color images.")
    sequence.add_argument("--depth-dir", type=Path, help="Directory of depth images aligned to color.")
    sequence.add_argument("--pose-file", type=Path, help="slambook pose.txt with tx ty tz qx qy qz qw rows.")
    sequence.add_argument("--color-pattern", default="*.png")
    sequence.add_argument("--depth-pattern", default="*.png")
    parser.add_argument("--output", required=True, type=Path, help="Output ASCII PLY path.")
    parser.add_argument("--intrinsics", required=True, type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc", type=float)
    parser.add_argument("--voxel-size", type=float, help="Optional voxel size for centroid downsampling.")
    parser.add_argument("--estimate-normals", action="store_true", help="Estimate PCA normals before writing PLY.")
    parser.add_argument("--normal-k", type=int, default=16, help="Neighbor count for --estimate-normals.")
    parser.add_argument("--mesh-output", type=Path, help="Optional Open3D Poisson mesh output path.")
    parser.add_argument("--poisson-depth", type=int, default=8, help="Poisson reconstruction depth for --mesh-output.")
    parser.add_argument("--occupancy-output", type=Path, help="Optional occupied voxel grid .npz output path.")
    parser.add_argument("--occupancy-voxel-size", type=float, default=0.05)
    parser.add_argument("--rerun", action="store_true", help="Log the output point cloud to Rerun.")
    parser.add_argument("--rerun-entity", default="world/rgbd_cloud", help="Rerun entity path for --rerun.")
    return parser.parse_args()


def _read_color(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not read color image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_depth(path: Path):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"Could not read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def _cloud_from_paths(
    color_path: Path,
    depth_path: Path,
    intrinsics: CameraIntrinsics,
    *,
    depth_scale: float,
    depth_trunc: float | None,
):
    return rgbd_to_point_cloud(
        _read_color(color_path),
        _read_depth(depth_path),
        intrinsics,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
    )


def _load_fused_cloud(args: argparse.Namespace, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray | None, int]:
    single_requested = args.color is not None or args.depth is not None
    sequence_requested = args.color_dir is not None or args.depth_dir is not None or args.pose_file is not None
    if single_requested == sequence_requested:
        raise SystemExit("Pass either --color/--depth or --color-dir/--depth-dir/--pose-file.")
    if single_requested:
        if args.color is None or args.depth is None:
            raise SystemExit("Single-frame mode requires both --color and --depth.")
        cloud = _cloud_from_paths(
            args.color,
            args.depth,
            intrinsics,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
        )
        return cloud.points, cloud.colors, 1

    if args.color_dir is None or args.depth_dir is None or args.pose_file is None:
        raise SystemExit("Sequence mode requires --color-dir, --depth-dir, and --pose-file.")
    color_frames = list_image_sequence(args.color_dir, pattern=args.color_pattern)
    depth_frames = list_image_sequence(args.depth_dir, pattern=args.depth_pattern)
    poses_wc = read_slambook_pose_file(args.pose_file)
    if not color_frames:
        raise SystemExit("No color frames matched --color-pattern.")
    if len(color_frames) != len(depth_frames):
        raise SystemExit(f"Color/depth frame count mismatch: {len(color_frames)} vs {len(depth_frames)}.")
    if len(poses_wc) != len(color_frames):
        raise SystemExit(f"Pose/frame count mismatch: {len(poses_wc)} vs {len(color_frames)}.")

    clouds = []
    for color_frame, depth_frame in zip(color_frames, depth_frames):
        cloud = _cloud_from_paths(
            color_frame.image_path,
            depth_frame.image_path,
            intrinsics,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
        )
        clouds.append((cloud.points, cloud.colors))
    points, colors = fuse_point_clouds(clouds, transforms_wb=poses_wc)
    return points, colors, len(clouds)


def main() -> None:
    args = _parse_args()
    fx, fy, cx, cy = args.intrinsics
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    points, colors, frame_count = _load_fused_cloud(args, intrinsics)
    input_point_count = len(points)
    if args.voxel_size is not None:
        points, colors = voxel_downsample(points, colors, voxel_size=args.voxel_size)
    normals = estimate_normals(points, k=args.normal_k, viewpoint=np.zeros(3)) if args.estimate_normals else None
    write_ply_ascii(args.output, points, colors, normals)
    if args.occupancy_output is not None:
        occupancy = occupancy_voxel_grid(points, voxel_size=args.occupancy_voxel_size)
        write_occupancy_npz(args.occupancy_output, occupancy)
    if args.mesh_output is not None:
        try:
            mesh, densities = reconstruct_mesh_poisson(
                points,
                colors,
                normals,
                depth=args.poisson_depth,
            )
            write_triangle_mesh_open3d(args.mesh_output, mesh)
        except OptionalVisualizationDependencyError as exc:
            raise SystemExit(str(exc)) from exc
    if args.rerun:
        try:
            rr = require_rerun()
            rr.init("slambook_rgbd_fusion", spawn=True)
            log_points_rerun(args.rerun_entity, points, colors)
        except OptionalVisualizationDependencyError as exc:
            raise SystemExit(str(exc)) from exc

    if args.color is not None:
        print(f"color: {args.color}")
        print(f"depth: {args.depth}")
    else:
        print(f"color dir: {args.color_dir}")
        print(f"depth dir: {args.depth_dir}")
        print(f"pose file: {args.pose_file}")
        print(f"frame count: {frame_count}")
    print(f"output: {args.output}")
    print(f"point count: {input_point_count}")
    if args.voxel_size is not None:
        print(f"downsampled point count: {len(points)}")
        print(f"voxel size: {args.voxel_size}")
    if args.estimate_normals:
        print(f"normal count: {len(normals)}")
        print(f"normal k: {args.normal_k}")
    if args.mesh_output is not None:
        print(f"mesh output: {args.mesh_output}")
        print(f"mesh density count: {len(densities)}")
        print(f"poisson depth: {args.poisson_depth}")
    if args.occupancy_output is not None:
        print(f"occupancy output: {args.occupancy_output}")
        print(f"occupancy voxel count: {len(occupancy.indices)}")
        print(f"occupancy voxel size: {args.occupancy_voxel_size}")
    if args.rerun:
        print(f"logged Rerun point cloud: {args.rerun_entity}")
    print(f"depth scale: {args.depth_scale}")
    if args.depth_trunc is not None:
        print(f"depth truncation: {args.depth_trunc}")


if __name__ == "__main__":
    main()
