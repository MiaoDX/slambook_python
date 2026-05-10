"""Create a point cloud from one RGB-D frame and write it as PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.mapping.pointcloud import estimate_normals, voxel_downsample, write_ply_ascii
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
    parser.add_argument("--color", required=True, type=Path, help="Path to an RGB/BGR color image.")
    parser.add_argument("--depth", required=True, type=Path, help="Path to a depth image aligned to color.")
    parser.add_argument("--output", required=True, type=Path, help="Output ASCII PLY path.")
    parser.add_argument("--intrinsics", required=True, type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc", type=float)
    parser.add_argument("--voxel-size", type=float, help="Optional voxel size for centroid downsampling.")
    parser.add_argument("--estimate-normals", action="store_true", help="Estimate PCA normals before writing PLY.")
    parser.add_argument("--normal-k", type=int, default=16, help="Neighbor count for --estimate-normals.")
    parser.add_argument("--mesh-output", type=Path, help="Optional Open3D Poisson mesh output path.")
    parser.add_argument("--poisson-depth", type=int, default=8, help="Poisson reconstruction depth for --mesh-output.")
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


def main() -> None:
    args = _parse_args()
    fx, fy, cx, cy = args.intrinsics
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    color = _read_color(args.color)
    depth = _read_depth(args.depth)
    cloud = rgbd_to_point_cloud(
        color,
        depth,
        intrinsics,
        depth_scale=args.depth_scale,
        depth_trunc=args.depth_trunc,
    )
    points = cloud.points
    colors = cloud.colors
    if args.voxel_size is not None:
        points, colors = voxel_downsample(points, colors, voxel_size=args.voxel_size)
    normals = estimate_normals(points, k=args.normal_k, viewpoint=np.zeros(3)) if args.estimate_normals else None
    write_ply_ascii(args.output, points, colors, normals)
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

    print(f"color: {args.color}")
    print(f"depth: {args.depth}")
    print(f"output: {args.output}")
    print(f"point count: {len(cloud.points)}")
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
    if args.rerun:
        print(f"logged Rerun point cloud: {args.rerun_entity}")
    print(f"depth scale: {args.depth_scale}")
    if args.depth_trunc is not None:
        print(f"depth truncation: {args.depth_trunc}")


if __name__ == "__main__":
    main()
