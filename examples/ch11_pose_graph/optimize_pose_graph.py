"""Optimize a `.g2o` SE3 pose graph with a small SciPy baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from slam.io.trajectory import write_kitti_trajectory, write_tum_trajectory
from slam.optimization.gtsam_backend import OptionalBackendDependencyError, optimize_pose_graph_gtsam
from slam.optimization.pose_graph import pose_graph_to_trajectory, read_g2o_pose_graph, solve_pose_graph
from slam.viz import OptionalVisualizationDependencyError, log_trajectory_rerun, require_rerun, save_trajectory_plot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2o", required=True, type=Path, help="Path to a .g2o pose graph.")
    parser.add_argument("--backend", choices=("scipy", "gtsam"), default="scipy")
    parser.add_argument("--fixed-vertex", type=int, help="Vertex id to hold fixed. Defaults to the smallest id.")
    parser.add_argument("--max-nfev", type=int, help="Maximum SciPy residual evaluations.")
    parser.add_argument("--max-iterations", type=int, help="Maximum GTSAM Levenberg-Marquardt iterations.")
    parser.add_argument("--output-tum", type=Path, help="Optional optimized TUM trajectory output path.")
    parser.add_argument("--output-kitti", type=Path, help="Optional optimized KITTI trajectory output path.")
    parser.add_argument("--plot-output", type=Path, help="Optional optimized trajectory plot image path.")
    parser.add_argument("--rerun", action="store_true", help="Log the optimized trajectory to Rerun.")
    parser.add_argument("--rerun-entity", default="world/optimized_trajectory", help="Rerun entity path for --rerun.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    graph = read_g2o_pose_graph(args.g2o)
    if args.backend == "scipy":
        result = solve_pose_graph(graph, fixed_vertex_id=args.fixed_vertex, max_nfev=args.max_nfev)
    else:
        try:
            result = optimize_pose_graph_gtsam(
                graph,
                fixed_vertex_id=args.fixed_vertex,
                max_iterations=args.max_iterations,
            )
        except OptionalBackendDependencyError as exc:
            raise SystemExit(str(exc)) from exc

    print(f"g2o file: {args.g2o}")
    print(f"backend: {args.backend}")
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
    if args.plot_output is not None:
        try:
            save_trajectory_plot(
                args.plot_output,
                [pose.transform_wc for pose in trajectory],
                label="optimized pose graph",
            )
        except OptionalVisualizationDependencyError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"wrote trajectory plot: {args.plot_output}")
    if args.rerun:
        try:
            rr = require_rerun()
            rr.init("slambook_pose_graph", spawn=True)
            log_trajectory_rerun(
                args.rerun_entity,
                [pose.transform_wc for pose in trajectory],
                color=[0, 128, 255],
                radius=0.02,
            )
        except OptionalVisualizationDependencyError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"logged Rerun trajectory: {args.rerun_entity}")


if __name__ == "__main__":
    main()
