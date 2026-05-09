import numpy as np

from slam.optimization.pose_graph import edge_error, read_g2o_pose_graph, total_edge_error


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
