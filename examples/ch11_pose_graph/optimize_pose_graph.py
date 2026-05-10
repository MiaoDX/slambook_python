"""Optimize a `.g2o` SE3 pose graph with a small SciPy baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from slam.io.trajectory import write_kitti_trajectory, write_tum_trajectory
from slam.optimization.pose_graph import pose_graph_to_trajectory, read_g2o_pose_graph, solve_pose_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2o", required=True, type=Path, help="Path to a .g2o pose graph.")
    parser.add_argument("--fixed-vertex", type=int, help="Vertex id to hold fixed. Defaults to the smallest id.")
    parser.add_argument("--max-nfev", type=int, help="Maximum SciPy residual evaluations.")
    parser.add_argument("--output-tum", type=Path, help="Optional optimized TUM trajectory output path.")
    parser.add_argument("--output-kitti", type=Path, help="Optional optimized KITTI trajectory output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    graph = read_g2o_pose_graph(args.g2o)
    result = solve_pose_graph(graph, fixed_vertex_id=args.fixed_vertex, max_nfev=args.max_nfev)

    print(f"g2o file: {args.g2o}")
    print(f"vertex count: {len(graph.vertices)}")
    print(f"edge count: {len(graph.edges)}")
    print(f"initial edge error: {result.initial_error:.9f}")
    print(f"final edge error: {result.final_error:.9f}")
    print(f"final cost: {result.cost:.9f}")
    print(f"function evaluations: {result.nfev}")
    print(f"success: {result.success}")
    print(f"message: {result.message}")

    trajectory = pose_graph_to_trajectory(result.graph)
    if args.output_tum is not None:
        write_tum_trajectory(args.output_tum, trajectory)
        print(f"wrote TUM trajectory: {args.output_tum}")
    if args.output_kitti is not None:
        write_kitti_trajectory(args.output_kitti, trajectory)
        print(f"wrote KITTI trajectory: {args.output_kitti}")


if __name__ == "__main__":
    main()
