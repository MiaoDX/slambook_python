import sys
import types

import cv2
import numpy as np

from slam.vo.pycolmap_backend import estimate_absolute_pose_pycolmap


def test_estimate_absolute_pose_pycolmap_returns_pnp_result(monkeypatch):
    fake_pycolmap = _install_fake_pycolmap(monkeypatch)
    points_3d = np.array(
        [
            [-1.0, -0.6, 4.0],
            [-0.2, -0.7, 4.5],
            [0.6, -0.3, 5.0],
            [1.0, 0.2, 5.5],
        ]
    )
    points_2d = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
    camera_matrix = np.array([[700.0, 0.0, 320.0], [0.0, 710.0, 240.0], [0.0, 0.0, 1.0]])
    rotation, _ = cv2.Rodrigues(np.array([0.03, -0.08, 0.02]))
    translation = np.array([0.3, -0.1, 0.4])
    fake_pycolmap.next_pose = np.column_stack([rotation, translation])

    result = estimate_absolute_pose_pycolmap(
        points_3d,
        points_2d,
        camera_matrix,
        image_size=(640, 480),
        estimation_options={"ransac": {"max_error": 4.0}},
        refinement_options={"max_num_iterations": 10},
    )

    np.testing.assert_allclose(result.rotation, rotation)
    np.testing.assert_allclose(result.translation.reshape(3), translation)
    np.testing.assert_array_equal(result.mask, [True, False, True, True])
    assert result.inlier_count == 3
    np.testing.assert_allclose(fake_pycolmap.camera.params, [700.0, 710.0, 320.0, 240.0])
    name, actual_points_2d, actual_points_3d, camera, kwargs = fake_pycolmap.calls[0]
    assert name == "estimate_and_refine_absolute_pose"
    np.testing.assert_allclose(actual_points_2d, points_2d)
    np.testing.assert_allclose(actual_points_3d, points_3d)
    assert camera is fake_pycolmap.camera
    assert kwargs == {
        "estimation_options": {"ransac": {"max_error": 4.0}},
        "refinement_options": {"max_num_iterations": 10},
        "return_covariance": False,
    }


def _install_fake_pycolmap(monkeypatch):
    fake = types.ModuleType("pycolmap")
    fake.calls = []
    fake.camera = None
    fake.next_pose = np.eye(3, 4, dtype=np.float64)

    class Camera:
        def __init__(self, *, model, width, height, params):
            self.model = model
            self.width = width
            self.height = height
            self.params = list(params)

        @staticmethod
        def create_from_model_name(_camera_id, model, focal, width, height):
            return Camera(
                model=model,
                width=width,
                height=height,
                params=[focal, focal, width / 2.0, height / 2.0],
            )

    class Rigid3d:
        def __init__(self, matrix):
            self._matrix = np.asarray(matrix, dtype=np.float64)

        def matrix(self):
            return self._matrix

    def estimate_and_refine_absolute_pose(points_2d, points_3d, camera, **kwargs):
        fake.camera = camera
        fake.calls.append(("estimate_and_refine_absolute_pose", points_2d, points_3d, camera, kwargs))
        return {
            "cam_from_world": Rigid3d(fake.next_pose),
            "inliers": np.array([True, False, True, True], dtype=bool),
        }

    fake.Camera = Camera
    fake.estimate_and_refine_absolute_pose = estimate_and_refine_absolute_pose
    monkeypatch.setitem(sys.modules, "pycolmap", fake)
    return fake
