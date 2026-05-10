from pathlib import Path

import numpy as np

from slam.optimization.bundle_adjustment import (
    BALObservation,
    BALProblem,
    pack_bal_parameters,
    project_bal_point,
    read_bal_problem,
    reprojection_residuals,
    reprojection_rmse,
    solve_bundle_adjustment,
    unpack_bal_parameters,
)


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


def test_pack_unpack_bal_parameters_round_trip_selected_blocks():
    problem = BALProblem(
        camera_params=np.arange(18, dtype=np.float64).reshape(2, 9),
        points_3d=np.arange(12, dtype=np.float64).reshape(4, 3),
        observations=[],
    )

    packed = pack_bal_parameters(problem, optimize_cameras=False, optimize_points=True)
    unpacked = unpack_bal_parameters(problem, packed, optimize_cameras=False, optimize_points=True)

    np.testing.assert_allclose(unpacked.camera_params, problem.camera_params)
    np.testing.assert_allclose(unpacked.points_3d, problem.points_3d)


def test_solve_bundle_adjustment_reduces_reprojection_rmse_with_fixed_camera():
    camera = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, -3.0, 120.0, 0.0, 0.0]])
    true_points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, -0.1, 1.2],
            [-0.3, 0.2, 1.4],
            [0.1, 0.25, 1.6],
        ]
    )
    observations = [
        BALObservation(camera_index=0, point_index=index, xy=project_bal_point(camera[0], point))
        for index, point in enumerate(true_points)
    ]
    perturbed_points = true_points + np.array(
        [
            [0.05, -0.04, 0.0],
            [-0.04, 0.03, 0.0],
            [0.03, -0.02, 0.0],
            [-0.03, 0.01, 0.0],
        ]
    )
    problem = BALProblem(camera_params=camera, points_3d=perturbed_points, observations=observations)

    result = solve_bundle_adjustment(problem, optimize_cameras=False, optimize_points=True, max_nfev=100)

    assert result.success
    assert result.final_rmse < result.initial_rmse
    assert result.final_rmse < 1e-6


def test_included_ch10_fixture_runs_scipy_bundle_adjustment():
    fixture = Path(__file__).resolve().parents[2] / "examples" / "ch10_bundle_adjustment" / "tiny_bal.txt"
    problem = read_bal_problem(fixture)

    result = solve_bundle_adjustment(problem, optimize_cameras=False, optimize_points=True, max_nfev=100)

    assert problem.camera_params.shape == (1, 9)
    assert problem.points_3d.shape == (4, 3)
    assert len(problem.observations) == 4
    assert result.success
    assert result.final_rmse < result.initial_rmse
    assert result.final_rmse < 1e-6
