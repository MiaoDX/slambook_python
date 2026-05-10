from pathlib import Path

import numpy as np

from slam.optimization.pose_graph import (
    PoseGraph,
    PoseGraphVertex,
    edge_error,
    pose_graph_to_trajectory,
    read_g2o_pose_graph,
    solve_pose_graph,
    total_edge_error,
)


def test_read_g2o_pose_graph_parses_vertices_edges_and_information(tmp_path):
    path = tmp_path / "tiny.g2o"
    information_upper = " ".join(["1"] * 21)
    path.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1",
                f"EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 {information_upper}",
            ]
        ),
        encoding="utf-8",
    )

    graph = read_g2o_pose_graph(path)

    assert sorted(graph.vertices) == [0, 1]
    assert len(graph.edges) == 1
    np.testing.assert_allclose(graph.edges[0].information, np.ones((6, 6)))


def test_pose_graph_edge_error_is_zero_when_measurement_matches_poses(tmp_path):
    path = tmp_path / "tiny.g2o"
    information_upper = " ".join(["1"] * 21)
    path.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 1 1 0 0 0 0 0 1",
                f"EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 {information_upper}",
            ]
        ),
        encoding="utf-8",
    )
    graph = read_g2o_pose_graph(path)

    np.testing.assert_allclose(edge_error(graph, graph.edges[0]), np.zeros(6), atol=1e-12)
    assert total_edge_error(graph) == 0.0


def test_pose_graph_edge_error_detects_inconsistent_translation(tmp_path):
    path = tmp_path / "tiny.g2o"
    information_upper = " ".join(["1"] * 21)
    path.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 1 2 0 0 0 0 0 1",
                f"EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 {information_upper}",
            ]
        ),
        encoding="utf-8",
    )
    graph = read_g2o_pose_graph(path)

    assert total_edge_error(graph) > 0.0


def test_pose_graph_to_trajectory_sorts_vertices_and_copies_transforms():
    transform0 = np.eye(4, dtype=np.float64)
    transform2 = np.eye(4, dtype=np.float64)
    transform2[0, 3] = 2.0
    graph = PoseGraph(
        vertices={
            2: PoseGraphVertex(id=2, transform_wi=transform2),
            0: PoseGraphVertex(id=0, transform_wi=transform0),
        },
        edges=[],
    )

    trajectory = pose_graph_to_trajectory(graph)

    assert [pose.timestamp for pose in trajectory] == [0.0, 2.0]
    np.testing.assert_allclose(trajectory[1].transform_wc[:3, 3], [2.0, 0.0, 0.0])
    trajectory[1].transform_wc[0, 3] = 99.0
    assert graph.vertices[2].transform_wi[0, 3] == 2.0


def test_solve_pose_graph_reduces_inconsistent_translation_error(tmp_path):
    path = tmp_path / "tiny.g2o"
    information_upper = " ".join(["1"] * 21)
    path.write_text(
        "\n".join(
            [
                "VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1",
                "VERTEX_SE3:QUAT 1 2 0 0 0 0 0 1",
                f"EDGE_SE3:QUAT 0 1 1 0 0 0 0 0 1 {information_upper}",
            ]
        ),
        encoding="utf-8",
    )
    graph = read_g2o_pose_graph(path)

    result = solve_pose_graph(graph, fixed_vertex_id=0)

    assert result.success
    assert result.final_error < result.initial_error
    assert result.final_error < 1e-12
    np.testing.assert_allclose(result.graph.vertices[1].transform_wi[:3, 3], [1.0, 0.0, 0.0], atol=1e-6)


def test_included_ch11_fixture_optimizes_with_scipy_baseline():
    fixture = Path(__file__).resolve().parents[2] / "examples" / "ch11_pose_graph" / "tiny_pose_graph.g2o"
    graph = read_g2o_pose_graph(fixture)

    result = solve_pose_graph(graph, fixed_vertex_id=0)

    assert sorted(graph.vertices) == [0, 1, 2]
    assert len(graph.edges) == 3
    assert result.success
    assert result.final_error < result.initial_error
    assert result.final_error < 1e-12
    np.testing.assert_allclose(result.graph.vertices[1].transform_wi[:3, 3], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.graph.vertices[2].transform_wi[:3, 3], [2.0, 0.0, 0.0], atol=1e-6)
