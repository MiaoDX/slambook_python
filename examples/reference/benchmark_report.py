"""Write JSON benchmark reports for trajectories, BAL files, or pose graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slam.evaluation.metrics import bal_reprojection_report, pose_graph_report, trajectory_report
from slam.io.trajectory import read_tum_trajectory
from slam.optimization.bundle_adjustment import read_bal_problem, solve_bundle_adjustment
from slam.optimization.pose_graph import read_g2o_pose_graph, solve_pose_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    trajectory = subparsers.add_parser("trajectory")
    trajectory.add_argument("--estimated-tum", required=True, type=Path)
    trajectory.add_argument("--reference-tum", required=True, type=Path)
    trajectory.add_argument("--no-align-origin", action="store_true")
    trajectory.add_argument("--output", required=True, type=Path)

    bal = subparsers.add_parser("bal")
    bal.add_argument("--bal", required=True, type=Path)
    bal.add_argument("--solve", action="store_true", help="Also run SciPy BA and include final metrics.")
    bal.add_argument("--fix-cameras", action="store_true")
    bal.add_argument("--fix-points", action="store_true")
    bal.add_argument("--max-nfev", type=int, default=50)
    bal.add_argument("--output", required=True, type=Path)

    pose_graph = subparsers.add_parser("pose-graph")
    pose_graph.add_argument("--g2o", required=True, type=Path)
    pose_graph.add_argument("--solve", action="store_true", help="Also run SciPy pose graph optimization.")
    pose_graph.add_argument("--fixed-vertex-id", type=int)
    pose_graph.add_argument("--max-nfev", type=int, default=50)
    pose_graph.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "trajectory":
        reports = [
            trajectory_report(
                read_tum_trajectory(args.estimated_tum),
                read_tum_trajectory(args.reference_tum),
                align_origin=not args.no_align_origin,
            )
        ]
    elif args.mode == "bal":
        problem = read_bal_problem(args.bal)
        reports = [bal_reprojection_report(problem, name="bal_initial")]
        if args.solve:
            result = solve_bundle_adjustment(
                problem,
                optimize_cameras=not args.fix_cameras,
                optimize_points=not args.fix_points,
                max_nfev=args.max_nfev,
            )
            reports.append(bal_reprojection_report(result.problem, name="bal_final"))
    elif args.mode == "pose-graph":
        graph = read_g2o_pose_graph(args.g2o)
        reports = [pose_graph_report(graph, name="pose_graph_initial")]
        if args.solve:
            result = solve_pose_graph(
                graph,
                fixed_vertex_id=args.fixed_vertex_id,
                max_nfev=args.max_nfev,
            )
            reports.append(pose_graph_report(result.graph, name="pose_graph_final"))
    else:
        raise AssertionError(args.mode)

    payload = {"reports": [report.as_dict() for report in reports]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote benchmark report: {args.output}")
    for report in reports:
        print(f"{report.name}: {report.metrics}")


if __name__ == "__main__":
    main()
