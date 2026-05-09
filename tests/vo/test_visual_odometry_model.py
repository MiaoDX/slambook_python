import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.geometry.transforms import make_transform
from slam.vo.visual_odometry import Camera, Frame, Map, MapPoint, VisualOdometryConfig


def test_camera_wrapper_projects_with_intrinsics():
    camera = Camera(CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0))
    points = np.array([[0.0, 0.0, 2.0], [1.0, -1.0, 2.0]])

    pixels = camera.camera_to_pixel(points)

    np.testing.assert_allclose(pixels, [[320.0, 240.0], [570.0, -10.0]])


def test_frame_pose_update_and_keyframe_marking():
    frame = Frame(id=1, timestamp=2.5, image=np.zeros((4, 5), dtype=np.uint8))
    pose = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))

    frame.set_pose(pose)
    frame.mark_keyframe()

    assert frame.is_keyframe
    np.testing.assert_allclose(frame.pose_wc, pose)


def test_map_point_observation_lifecycle():
    point = MapPoint(id=7, position_w=np.array([1.0, 2.0, 3.0]))

    point.add_observation(frame_id=3, pixel=np.array([10.0, 20.0]))
    point.add_observation(frame_id=4, pixel=np.array([11.0, 21.0]))
    point.remove_observation(frame_id=3)

    assert point.observed_times == 1
    np.testing.assert_allclose(point.observations[4], [11.0, 21.0])


def test_map_inserts_keyframes_and_landmark_observations():
    slam_map = Map()
    frame = Frame(id=1, timestamp=0.0, image=np.zeros((2, 2), dtype=np.uint8))
    point = MapPoint(id=2, position_w=np.array([0.0, 0.0, 5.0]))

    slam_map.insert_keyframe(frame)
    slam_map.insert_map_point(point)
    slam_map.add_observation(point_id=2, frame_id=1, pixel=np.array([100.0, 120.0]))

    assert slam_map.keyframes[1].is_keyframe
    assert slam_map.points[2].observed_times == 1


def test_visual_odometry_config_defaults_are_core_only():
    config = VisualOdometryConfig()

    assert config.matcher == "orb"
    assert config.max_features > 0
    assert config.min_matches > 0


def test_visual_odometry_config_loads_from_yaml(tmp_path):
    path = tmp_path / "vo.yaml"
    path.write_text(
        "\n".join(
            [
                "matcher: sift",
                "max_features: 2000",
                "min_matches: 40",
                "min_pnp_inliers: 18",
                "keyframe_min_translation: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    config = VisualOdometryConfig.from_yaml(path)

    assert config.matcher == "sift"
    assert config.max_features == 2000
    assert config.min_matches == 40
    assert config.min_pnp_inliers == 18
    assert config.keyframe_min_translation == 0.25


def test_visual_odometry_config_rejects_unknown_fields():
    try:
        VisualOdometryConfig.from_mapping({"matcher": "orb", "surprise": True})
    except ValueError as exc:
        assert "surprise" in str(exc)
    else:
        raise AssertionError("expected ValueError")
