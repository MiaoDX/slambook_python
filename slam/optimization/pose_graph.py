"""Pose graph parsing and residual helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from slam.geometry.lie import se3_exp, se3_log
from slam.geometry.transforms import inverse_transform, make_transform
from slam.io.trajectory import PoseStamped


@dataclass(frozen=True)
class PoseGraphVertex:
    """One SE3 vertex pose as `T_wi`."""

    id: int
    transform_wi: np.ndarray


@dataclass(frozen=True)
class PoseGraphEdge:
    """Relative SE3 measurement `T_ij` between two vertices."""

    from_id: int
    to_id: int
    measurement_ij: np.ndarray
    information: np.ndarray


@dataclass(frozen=True)
class PoseGraph:
    vertices: dict[int, PoseGraphVertex]
    edges: list[PoseGraphEdge]


@dataclass(frozen=True)
class PoseGraphOptimizationResult:
    """Result from baseline SciPy pose graph optimization."""

    graph: PoseGraph
    initial_error: float
    final_error: float
    cost: float
    nfev: int
    success: bool
    message: str


def read_g2o_pose_graph(path: str | Path) -> PoseGraph:
    """Read `VERTEX_SE3:QUAT` and `EDGE_SE3:QUAT` records from a `.g2o` file."""

    vertices: dict[int, PoseGraphVertex] = {}
    edges: list[PoseGraphEdge] = []

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        tag = parts[0]
        if tag == "VERTEX_SE3:QUAT":
            if len(parts) != 9:
                raise ValueError(f"line {line_number}: VERTEX_SE3:QUAT expects 8 values")
            vertex_id = int(parts[1])
            x, y, z, qx, qy, qz, qw = (float(value) for value in parts[2:])
            vertices[vertex_id] = PoseGraphVertex(
                id=vertex_id,
                transform_wi=_transform_from_xyz_quat(x, y, z, qx, qy, qz, qw),
            )
        elif tag == "EDGE_SE3:QUAT":
            if len(parts) != 31:
                raise ValueError(f"line {line_number}: EDGE_SE3:QUAT expects 30 values")
            from_id = int(parts[1])
            to_id = int(parts[2])
            x, y, z, qx, qy, qz, qw = (float(value) for value in parts[3:10])
            information_values = [float(value) for value in parts[10:]]
            edges.append(
                PoseGraphEdge(
                    from_id=from_id,
                    to_id=to_id,
                    measurement_ij=_transform_from_xyz_quat(x, y, z, qx, qy, qz, qw),
                    information=_upper_triangle_to_matrix(information_values, size=6),
                )
            )
    return PoseGraph(vertices=vertices, edges=edges)


def edge_error(graph: PoseGraph, edge: PoseGraphEdge) -> np.ndarray:
    """Return 6D SE3 log error for one edge."""

    from_pose = graph.vertices[edge.from_id].transform_wi
    to_pose = graph.vertices[edge.to_id].transform_wi
    predicted_ij = inverse_transform(from_pose) @ to_pose
    error_transform = inverse_transform(edge.measurement_ij) @ predicted_ij
    return se3_log(error_transform)


def total_edge_error(graph: PoseGraph) -> float:
    """Return sum of squared unweighted SE3 edge errors."""

    total = 0.0
    for edge in graph.edges:
        error = edge_error(graph, edge)
        total += float(error @ error)
    return total


def pose_graph_to_trajectory(graph: PoseGraph) -> list[PoseStamped]:
    """Return graph vertices as a sorted timestamped trajectory.

    Vertex ids become timestamps, and each `T_wi` is exported as the world pose
    for trajectory writers such as TUM and KITTI.
    """

    return [
        PoseStamped(timestamp=float(vertex_id), transform_wc=graph.vertices[vertex_id].transform_wi.copy())
        for vertex_id in sorted(graph.vertices)
    ]


def pack_pose_graph_parameters(graph: PoseGraph, *, fixed_vertex_id: int | None = None) -> np.ndarray:
    """Pack non-fixed vertex poses as SE3 log vectors."""

    ids = _optimized_vertex_ids(graph, fixed_vertex_id=fixed_vertex_id)
    if not ids:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([se3_log(graph.vertices[vertex_id].transform_wi) for vertex_id in ids])


def unpack_pose_graph_parameters(
    graph: PoseGraph,
    params: np.ndarray,
    *,
    fixed_vertex_id: int | None = None,
) -> PoseGraph:
    """Unpack SE3 log vectors into a new graph, preserving fixed vertices and edges."""

    params = np.asarray(params, dtype=np.float64).reshape(-1)
    ids = _optimized_vertex_ids(graph, fixed_vertex_id=fixed_vertex_id)
    expected = 6 * len(ids)
    if len(params) != expected:
        raise ValueError(f"expected {expected} pose graph parameters, got {len(params)}")

    vertices = dict(graph.vertices)
    for index, vertex_id in enumerate(ids):
        vertices[vertex_id] = PoseGraphVertex(
            id=vertex_id,
            transform_wi=se3_exp(params[6 * index : 6 * index + 6]),
        )
    return PoseGraph(vertices=vertices, edges=graph.edges)


def pose_graph_residuals_from_parameters(
    params: np.ndarray,
    graph: PoseGraph,
    *,
    fixed_vertex_id: int | None = None,
) -> np.ndarray:
    """Flatten all pose graph edge errors for a packed parameter vector."""

    unpacked = unpack_pose_graph_parameters(graph, params, fixed_vertex_id=fixed_vertex_id)
    if not unpacked.edges:
        return np.empty(0, dtype=np.float64)
    return np.concatenate([edge_error(unpacked, edge) for edge in unpacked.edges])


def solve_pose_graph(
    graph: PoseGraph,
    *,
    fixed_vertex_id: int | None = None,
    max_nfev: int | None = None,
) -> PoseGraphOptimizationResult:
    """Optimize pose graph vertices with SciPy least-squares."""

    if fixed_vertex_id is None and graph.vertices:
        fixed_vertex_id = min(graph.vertices)

    initial_params = pack_pose_graph_parameters(graph, fixed_vertex_id=fixed_vertex_id)
    initial_error = total_edge_error(graph)
    if initial_params.size == 0:
        return PoseGraphOptimizationResult(
            graph=graph,
            initial_error=initial_error,
            final_error=initial_error,
            cost=0.0,
            nfev=0,
            success=True,
            message="No vertices selected for optimization.",
        )

    result = least_squares(
        pose_graph_residuals_from_parameters,
        initial_params,
        args=(graph,),
        kwargs={"fixed_vertex_id": fixed_vertex_id},
        max_nfev=max_nfev,
    )
    optimized = unpack_pose_graph_parameters(graph, result.x, fixed_vertex_id=fixed_vertex_id)
    return PoseGraphOptimizationResult(
        graph=optimized,
        initial_error=initial_error,
        final_error=total_edge_error(optimized),
        cost=float(result.cost),
        nfev=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
    )


def _transform_from_xyz_quat(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return make_transform(rotation, np.array([x, y, z], dtype=np.float64))


def _optimized_vertex_ids(graph: PoseGraph, *, fixed_vertex_id: int | None) -> list[int]:
    return [vertex_id for vertex_id in sorted(graph.vertices) if vertex_id != fixed_vertex_id]


def _upper_triangle_to_matrix(values: list[float], *, size: int) -> np.ndarray:
    expected = size * (size + 1) // 2
    if len(values) != expected:
        raise ValueError(f"expected {expected} upper-triangle values, got {len(values)}")
    matrix = np.zeros((size, size), dtype=np.float64)
    cursor = 0
    for row in range(size):
        for col in range(row, size):
            matrix[row, col] = values[cursor]
            matrix[col, row] = values[cursor]
            cursor += 1
    return matrix
