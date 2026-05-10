import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.geometry.transforms import make_transform
from slam.vo.visual_odometry import (
    Camera,
    Frame,
    LocalMapMatchSet,
    Map,
    MapPoint,
    VisualOdometry,
    VisualOdometryConfig,
    chain_relative_pose,
    create_frame,
    estimate_frame_pose_from_local_map,
    extract_frame_features,
    match_local_map,
)
from slam.vo.pnp import project_points


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


def test_extract_frame_features_returns_point_array_for_blank_image():
    image = np.zeros((24, 24), dtype=np.uint8)

    keypoints, descriptors = extract_frame_features(image, max_features=50)

    assert keypoints.shape == (0, 2)
    assert descriptors is None


def test_create_frame_populates_features_and_pose():
    image = np.zeros((24, 24), dtype=np.uint8)
    pose = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))

    frame = create_frame(4, 1.25, image, pose_wc=pose, config=VisualOdometryConfig(max_features=50))

    assert frame.id == 4
    assert frame.timestamp == 1.25
    assert frame.keypoints.shape == (0, 2)
    np.testing.assert_allclose(frame.pose_wc, pose)


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


def test_estimate_frame_pose_from_local_map_tracks_pose_with_pnp_ransac():
    camera = Camera(CameraIntrinsics(fx=700.0, fy=710.0, cx=320.0, cy=240.0))
    points_w = np.array(
        [
            [-1.0, -0.6, 4.0],
            [-0.2, -0.7, 4.5],
            [0.6, -0.3, 5.0],
            [1.0, 0.2, 5.5],
            [-0.8, 0.4, 6.0],
            [0.0, 0.7, 6.5],
        ]
    )
    descriptors = np.zeros((len(points_w), 32), dtype=np.uint8)
    descriptors[:, 0] = np.arange(1, len(points_w) + 1, dtype=np.uint8)
    slam_map = Map(
        points={
            index: MapPoint(id=index, position_w=point_w, descriptor=descriptors[index])
            for index, point_w in enumerate(points_w)
        }
    )
    transform_cw = make_transform(np.eye(3), np.array([0.2, -0.1, 0.3]))
    pixels = project_points(points_w, transform_cw[:3, :3], transform_cw[:3, 3], camera.matrix)
    frame = Frame(
        id=2,
        timestamp=0.1,
        image=np.zeros((8, 8), dtype=np.uint8),
        keypoints=pixels,
        descriptors=descriptors.copy(),
    )
    config = VisualOdometryConfig(min_pnp_inliers=4)

    result = estimate_frame_pose_from_local_map(frame, slam_map, camera, config=config)

    assert result.success
    assert result.pnp is not None
    assert result.pnp.inlier_count >= 4
    np.testing.assert_allclose(result.pose_wc, np.linalg.inv(transform_cw), atol=1e-5)


def test_estimate_frame_pose_from_local_map_reports_insufficient_matches():
    camera = Camera(CameraIntrinsics(fx=700.0, fy=710.0, cx=320.0, cy=240.0))
    frame = Frame(id=2, timestamp=0.1, image=np.zeros((8, 8), dtype=np.uint8))

    result = estimate_frame_pose_from_local_map(frame, Map(), camera)

    assert not result.success
    assert result.pose_wc is None
    assert result.pnp is None
    assert "need at least 4" in result.message


def test_visual_odometry_keyframe_threshold_uses_last_keyframe_pose():
    camera = Camera(CameraIntrinsics(fx=700.0, fy=710.0, cx=320.0, cy=240.0))
    vo = VisualOdometry(camera=camera, config=VisualOdometryConfig(keyframe_min_translation=0.2))
    first = Frame(id=1, timestamp=0.0, image=np.zeros((4, 4), dtype=np.uint8))
    near = Frame(
        id=2,
        timestamp=0.1,
        image=np.zeros((4, 4), dtype=np.uint8),
        pose_wc=make_transform(np.eye(3), np.array([0.1, 0.0, 0.0])),
    )
    far = Frame(
        id=3,
        timestamp=0.2,
        image=np.zeros((4, 4), dtype=np.uint8),
        pose_wc=make_transform(np.eye(3), np.array([0.3, 0.0, 0.0])),
    )

    assert vo.should_insert_keyframe(first)
    vo.insert_keyframe(first)

    assert not vo.should_insert_keyframe(near)
    assert vo.should_insert_keyframe(far)


def test_visual_odometry_create_frame_uses_config():
    camera = Camera(CameraIntrinsics(fx=700.0, fy=710.0, cx=320.0, cy=240.0))
    vo = VisualOdometry(camera=camera, config=VisualOdometryConfig(max_features=25))

    frame = vo.create_frame(5, 2.0, np.zeros((24, 24), dtype=np.uint8))

    assert frame.id == 5
    assert frame.keypoints.shape == (0, 2)


def test_visual_odometry_track_local_map_updates_pose_and_keyframes():
    camera = Camera(CameraIntrinsics(fx=700.0, fy=710.0, cx=320.0, cy=240.0))
    config = VisualOdometryConfig(min_pnp_inliers=4, keyframe_min_translation=0.05)
    vo = VisualOdometry(camera=camera, config=config)
    vo.insert_keyframe(Frame(id=0, timestamp=0.0, image=np.zeros((8, 8), dtype=np.uint8)))
    points_w = np.array(
        [
            [-1.0, -0.6, 4.0],
            [-0.2, -0.7, 4.5],
            [0.6, -0.3, 5.0],
            [1.0, 0.2, 5.5],
            [-0.8, 0.4, 6.0],
            [0.0, 0.7, 6.5],
        ]
    )
    descriptors = np.zeros((len(points_w), 32), dtype=np.uint8)
    descriptors[:, 0] = np.arange(1, len(points_w) + 1, dtype=np.uint8)
    for index, point_w in enumerate(points_w):
        vo.map.insert_map_point(MapPoint(id=index, position_w=point_w, descriptor=descriptors[index]))
    transform_cw = make_transform(np.eye(3), np.array([0.2, -0.1, 0.3]))
    frame = Frame(
        id=2,
        timestamp=0.1,
        image=np.zeros((8, 8), dtype=np.uint8),
        keypoints=project_points(points_w, transform_cw[:3, :3], transform_cw[:3, 3], camera.matrix),
        descriptors=descriptors.copy(),
    )

    result = vo.track_local_map(frame)

    assert result.success
    assert vo.current_frame is frame
    assert frame.id in vo.map.keyframes
    np.testing.assert_allclose(frame.pose_wc, np.linalg.inv(transform_cw), atol=1e-5)


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
