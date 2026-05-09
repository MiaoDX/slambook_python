"""Pose graph parsing and residual helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from slam.geometry.lie import se3_log
from slam.geometry.transforms import inverse_transform, make_transform


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


def _transform_from_xyz_quat(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    return make_transform(rotation, np.array([x, y, z], dtype=np.float64))


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
