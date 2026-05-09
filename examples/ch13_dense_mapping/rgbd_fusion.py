"""Create a point cloud from one RGB-D frame and write it as PLY."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from slam.camera.pinhole import CameraIntrinsics
from slam.mapping.pointcloud import write_ply_ascii
from slam.mapping.rgbd import rgbd_to_point_cloud


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", required=True, type=Path, help="Path to an RGB/BGR color image.")
    parser.add_argument("--depth", required=True, type=Path, help="Path to a depth image aligned to color.")
    parser.add_argument("--output", required=True, type=Path, help="Output ASCII PLY path.")
    parser.add_argument("--intrinsics", required=True, type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--depth-trunc", type=float)
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
    write_ply_ascii(args.output, cloud.points, cloud.colors)

    print(f"color: {args.color}")
    print(f"depth: {args.depth}")
    print(f"output: {args.output}")
    print(f"point count: {len(cloud.points)}")
    print(f"depth scale: {args.depth_scale}")
    if args.depth_trunc is not None:
        print(f"depth truncation: {args.depth_trunc}")


if __name__ == "__main__":
    main()
