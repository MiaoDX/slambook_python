import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.geometry.transforms import make_transform
from slam.vo.visual_odometry import (
    Camera,
    Frame,
    LocalMapMatchSet,
    Map,
    MapPoint,
    VisualOdometryConfig,
    chain_relative_pose,
    match_local_map,
)


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


def test_match_local_map_returns_pnp_correspondences():
    zero_descriptor = np.zeros(32, dtype=np.uint8)
    full_descriptor = np.full(32, 255, dtype=np.uint8)
    slam_map = Map(
        points={
            7: MapPoint(id=7, position_w=np.array([1.0, 0.0, 4.0]), descriptor=zero_descriptor),
            8: MapPoint(id=8, position_w=np.array([0.0, 1.0, 5.0]), descriptor=full_descriptor),
        }
    )
    frame = Frame(
        id=2,
        timestamp=0.1,
        image=np.zeros((8, 8), dtype=np.uint8),
        keypoints=np.array([[40.0, 50.0], [10.0, 20.0]]),
        descriptors=np.vstack([full_descriptor, zero_descriptor]),
    )

    matches = match_local_map(frame, slam_map)
    pixel_by_point_id = {point_id: pixel for point_id, pixel in zip(matches.point_ids, matches.points_2d)}

    assert len(matches) == 2
    np.testing.assert_allclose(pixel_by_point_id[7], [10.0, 20.0])
    np.testing.assert_allclose(pixel_by_point_id[8], [40.0, 50.0])
    assert np.all(matches.distances == 0.0)


def test_match_local_map_returns_empty_without_descriptors():
    slam_map = Map(points={1: MapPoint(id=1, position_w=np.array([0.0, 0.0, 1.0]))})
    frame = Frame(id=2, timestamp=0.0, image=np.zeros((4, 4), dtype=np.uint8))

    matches = match_local_map(frame, slam_map)

    assert len(matches) == 0
    assert matches.points_3d.shape == (0, 3)
    assert matches.points_2d.shape == (0, 2)


def test_local_map_match_set_rejects_mismatched_lengths():
    try:
        LocalMapMatchSet(
            point_ids=np.array([1, 2]),
            points_3d=np.zeros((1, 3)),
            points_2d=np.zeros((1, 2)),
            frame_keypoint_indices=np.array([0]),
            distances=np.array([0.0]),
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


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


def test_chain_relative_pose_converts_t10_to_next_world_pose():
    previous_pose = np.eye(4)
    rotation_10 = np.eye(3)
    translation_10 = np.array([1.0, 0.0, 0.0])

    next_pose = chain_relative_pose(previous_pose, rotation_10, translation_10, translation_scale=2.0)

    np.testing.assert_allclose(next_pose[:3, 3], [-2.0, 0.0, 0.0])
