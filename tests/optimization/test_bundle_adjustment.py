import numpy as np

from slam.optimization.bundle_adjustment import BALObservation, BALProblem, read_bal_problem, reprojection_residuals, reprojection_rmse


def test_read_bal_problem_parses_counts_and_arrays(tmp_path):
    path = tmp_path / "tiny.bal"
    path.write_text(
        "\n".join(
            [
                "1 2 2",
                "0 0 10.0 20.0",
                "0 1 30.0 40.0",
                "0 0 0 0 0 -1 100 0 0",
                "0.1 0.2 1.0",
                "0.3 0.4 1.0",
            ]
        ),
        encoding="utf-8",
    )

    problem = read_bal_problem(path)

    assert problem.camera_params.shape == (1, 9)
    assert problem.points_3d.shape == (2, 3)
    assert len(problem.observations) == 2
    np.testing.assert_allclose(problem.observations[1].xy, [30.0, 40.0])


def test_reprojection_residuals_are_zero_for_matching_observation():
    camera = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 100.0, 0.0, 0.0]])
    point = np.array([[0.1, 0.2, 1.0]])
    problem = BALProblem(
        camera_params=camera,
        points_3d=point,
        observations=[BALObservation(camera_index=0, point_index=0, xy=np.array([10.0, 20.0]))],
    )

    residuals = reprojection_residuals(problem)

    np.testing.assert_allclose(residuals, [0.0, 0.0], atol=1e-12)
    assert reprojection_rmse(problem) == 0.0


def test_reprojection_residual_vector_shape_is_two_per_observation():
    problem = BALProblem(
        camera_params=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 100.0, 0.0, 0.0]]),
        points_3d=np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]),
        observations=[
            BALObservation(camera_index=0, point_index=0, xy=np.array([0.0, 0.0])),
            BALObservation(camera_index=0, point_index=1, xy=np.array([1.0, 0.0])),
        ],
    )

    assert reprojection_residuals(problem).shape == (4,)
