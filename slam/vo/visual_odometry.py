"""Data model for the Chapter 9 mini visual odometry project."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

from slam.camera.pinhole import CameraIntrinsics
from slam.features.opencv_features import match_descriptors
from slam.geometry.transforms import inverse_transform, make_transform
from slam.vo.pnp import PnPResult, solve_pnp_ransac


@dataclass(frozen=True)
class Camera:
    """Pinhole camera wrapper used by the mini VO project."""

    intrinsics: CameraIntrinsics

    @property
    def matrix(self) -> np.ndarray:
        return self.intrinsics.matrix

    def pixel_to_camera(self, pixels: np.ndarray, depths: np.ndarray | float = 1.0) -> np.ndarray:
        return self.intrinsics.pixel_to_camera(pixels, depths)

    def camera_to_pixel(self, points: np.ndarray) -> np.ndarray:
        return self.intrinsics.camera_to_pixel(points)


@dataclass
class Frame:
    """Image frame with pose and optional feature data."""

    id: int
    timestamp: float
    image: np.ndarray
    pose_wc: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    keypoints: np.ndarray | None = None
    descriptors: np.ndarray | None = None
    is_keyframe: bool = False

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        self.pose_wc = np.asarray(self.pose_wc, dtype=np.float64)
        if self.pose_wc.shape != (4, 4):
            raise ValueError("pose_wc must have shape 4x4")
        if self.keypoints is not None:
            self.keypoints = _points2(self.keypoints, name="keypoints")

    def set_pose(self, pose_wc: np.ndarray) -> None:
        pose_wc = np.asarray(pose_wc, dtype=np.float64)
        if pose_wc.shape != (4, 4):
            raise ValueError("pose_wc must have shape 4x4")
        self.pose_wc = pose_wc

    def mark_keyframe(self) -> None:
        self.is_keyframe = True


@dataclass
class MapPoint:
    """3D landmark with frame observations."""

    id: int
    position_w: np.ndarray
    descriptor: np.ndarray | None = None
    observations: dict[int, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position_w = np.asarray(self.position_w, dtype=np.float64).reshape(3)
        if self.descriptor is not None:
            self.descriptor = np.asarray(self.descriptor)

    @property
    def observed_times(self) -> int:
        return len(self.observations)

    def add_observation(self, frame_id: int, pixel: np.ndarray) -> None:
        self.observations[int(frame_id)] = np.asarray(pixel, dtype=np.float64).reshape(2)

    def remove_observation(self, frame_id: int) -> None:
        self.observations.pop(int(frame_id), None)


@dataclass(frozen=True)
class LocalMapMatchSet:
    """3D/2D correspondences between map points and one frame."""

    point_ids: np.ndarray
    points_3d: np.ndarray
    points_2d: np.ndarray
    frame_keypoint_indices: np.ndarray
    distances: np.ndarray

    def __post_init__(self) -> None:
        point_ids = np.asarray(self.point_ids, dtype=np.int64).reshape(-1)
        points_3d = np.asarray(self.points_3d, dtype=np.float64).reshape(-1, 3)
        points_2d = np.asarray(self.points_2d, dtype=np.float64).reshape(-1, 2)
        frame_keypoint_indices = np.asarray(self.frame_keypoint_indices, dtype=np.int64).reshape(-1)
        distances = np.asarray(self.distances, dtype=np.float64).reshape(-1)
        sizes = {len(point_ids), len(points_3d), len(points_2d), len(frame_keypoint_indices), len(distances)}
        if len(sizes) != 1:
            raise ValueError("local map match arrays must have the same length")
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "points_3d", points_3d)
        object.__setattr__(self, "points_2d", points_2d)
        object.__setattr__(self, "frame_keypoint_indices", frame_keypoint_indices)
        object.__setattr__(self, "distances", distances)

    def __len__(self) -> int:
        return len(self.point_ids)


@dataclass(frozen=True)
class LocalMapTrackingResult:
    """Pose tracking result from matching a frame against the local map."""

    success: bool
    pose_wc: np.ndarray | None
    pnp: PnPResult | None
    matches: LocalMapMatchSet
    message: str = ""

    def __post_init__(self) -> None:
        if self.pose_wc is None:
            return
        pose_wc = np.asarray(self.pose_wc, dtype=np.float64)
        if pose_wc.shape != (4, 4):
            raise ValueError("pose_wc must have shape 4x4")
        object.__setattr__(self, "pose_wc", pose_wc)


@dataclass
class Map:
    """Container for keyframes and landmarks."""

    keyframes: dict[int, Frame] = field(default_factory=dict)
    points: dict[int, MapPoint] = field(default_factory=dict)

    def insert_keyframe(self, frame: Frame) -> None:
        frame.mark_keyframe()
        self.keyframes[frame.id] = frame

    def insert_map_point(self, point: MapPoint) -> None:
        self.points[point.id] = point

    def add_observation(self, point_id: int, frame_id: int, pixel: np.ndarray) -> None:
        if point_id not in self.points:
            raise KeyError(f"unknown map point id {point_id}")
        if frame_id not in self.keyframes:
            raise KeyError(f"unknown keyframe id {frame_id}")
        self.points[point_id].add_observation(frame_id, pixel)

    def remove_map_point(self, point_id: int) -> None:
        self.points.pop(point_id, None)


@dataclass(frozen=True)
class VisualOdometryConfig:
    """Configuration values for the mini VO pipeline."""

    matcher: str = "orb"
    max_features: int = 1000
    min_matches: int = 20
    min_pnp_inliers: int = 10
    keyframe_min_translation: float = 0.1

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "VisualOdometryConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown VisualOdometryConfig fields: {', '.join(unknown)}")
        return cls(**{key: values[key] for key in allowed if key in values})

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VisualOdometryConfig":
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError("VO config YAML must contain a mapping")
        return cls.from_mapping(data)


def chain_relative_pose(
    previous_pose_wc: np.ndarray,
    rotation_10: np.ndarray,
    translation_10: np.ndarray,
    *,
    translation_scale: float = 1.0,
) -> np.ndarray:
    """Compose a recovered two-view pose into a world-camera trajectory.

    `rotation_10` and `translation_10` are interpreted as `T_10`, mapping
    camera 0 coordinates into camera 1 coordinates. Returned pose is `T_wc1`.
    """

    previous_pose_wc = np.asarray(previous_pose_wc, dtype=np.float64)
    if previous_pose_wc.shape != (4, 4):
        raise ValueError("previous_pose_wc must have shape 4x4")

    transform_10 = make_transform(rotation_10, np.asarray(translation_10, dtype=np.float64).reshape(3) * translation_scale)
    transform_01 = inverse_transform(transform_10)
    return previous_pose_wc @ transform_01


def match_local_map(
    frame: Frame,
    slam_map: Map,
    *,
    feature: str = "orb",
    max_matches: int | None = None,
) -> LocalMapMatchSet:
    """Match frame descriptors against map point descriptors for PnP tracking."""

    if frame.keypoints is None or frame.descriptors is None:
        return _empty_local_map_matches()

    point_ids: list[int] = []
    descriptor_rows = []
    for point_id, point in sorted(slam_map.points.items()):
        if point.descriptor is None:
            continue
        point_ids.append(point_id)
        descriptor_rows.append(np.asarray(point.descriptor).reshape(1, -1))

    if not descriptor_rows:
        return _empty_local_map_matches()

    map_descriptors = np.vstack(descriptor_rows)
    frame_descriptors = np.asarray(frame.descriptors)
    if map_descriptors.dtype != frame_descriptors.dtype:
        map_descriptors = map_descriptors.astype(frame_descriptors.dtype, copy=False)

    matches = match_descriptors(map_descriptors, frame_descriptors, feature=feature)
    if max_matches is not None:
        matches = matches[:max_matches]
    if not matches:
        return _empty_local_map_matches()

    matched_point_ids = []
    points_3d = []
    points_2d = []
    frame_keypoint_indices = []
    distances = []
    for match in matches:
        point_id = point_ids[match.queryIdx]
        frame_index = int(match.trainIdx)
        matched_point_ids.append(point_id)
        points_3d.append(slam_map.points[point_id].position_w)
        points_2d.append(frame.keypoints[frame_index])
        frame_keypoint_indices.append(frame_index)
        distances.append(float(match.distance))

    return LocalMapMatchSet(
        point_ids=np.asarray(matched_point_ids, dtype=np.int64),
        points_3d=np.asarray(points_3d, dtype=np.float64),
        points_2d=np.asarray(points_2d, dtype=np.float64),
        frame_keypoint_indices=np.asarray(frame_keypoint_indices, dtype=np.int64),
        distances=np.asarray(distances, dtype=np.float64),
    )


def estimate_frame_pose_from_local_map(
    frame: Frame,
    slam_map: Map,
    camera: Camera,
    *,
    config: VisualOdometryConfig | None = None,
    feature: str | None = None,
    max_matches: int | None = None,
) -> LocalMapTrackingResult:
    """Estimate a frame world pose from local map matches and PnP RANSAC."""

    config = VisualOdometryConfig() if config is None else config
    matches = match_local_map(
        frame,
        slam_map,
        feature=feature or config.matcher,
        max_matches=max_matches,
    )
    if len(matches) < 4:
        return LocalMapTrackingResult(
            success=False,
            pose_wc=None,
            pnp=None,
            matches=matches,
            message=f"need at least 4 local map matches, got {len(matches)}",
        )

    try:
        pnp = solve_pnp_ransac(matches.points_3d, matches.points_2d, camera.matrix)
    except RuntimeError as exc:
        return LocalMapTrackingResult(success=False, pose_wc=None, pnp=None, matches=matches, message=str(exc))

    if pnp.inlier_count < config.min_pnp_inliers:
        return LocalMapTrackingResult(
            success=False,
            pose_wc=None,
            pnp=pnp,
            matches=matches,
            message=f"need at least {config.min_pnp_inliers} PnP inliers, got {pnp.inlier_count}",
        )

    transform_cw = make_transform(pnp.rotation, pnp.translation)
    pose_wc = inverse_transform(transform_cw)
    return LocalMapTrackingResult(success=True, pose_wc=pose_wc, pnp=pnp, matches=matches)


@dataclass
class VisualOdometry:
    """Small stateful coordinator for the Chapter 9 mini VO project."""

    camera: Camera
    config: VisualOdometryConfig = field(default_factory=VisualOdometryConfig)
    map: Map = field(default_factory=Map)
    current_frame: Frame | None = None

    def insert_keyframe(self, frame: Frame) -> Frame:
        self.map.insert_keyframe(frame)
        self.current_frame = frame
        return frame

    def should_insert_keyframe(self, frame: Frame) -> bool:
        if not self.map.keyframes:
            return True
        latest_keyframe = self.map.keyframes[max(self.map.keyframes)]
        translation_delta = np.linalg.norm(frame.pose_wc[:3, 3] - latest_keyframe.pose_wc[:3, 3])
        return bool(translation_delta >= self.config.keyframe_min_translation)

    def track_local_map(
        self,
        frame: Frame,
        *,
        feature: str | None = None,
        max_matches: int | None = None,
        insert_keyframe: bool = True,
    ) -> LocalMapTrackingResult:
        result = estimate_frame_pose_from_local_map(
            frame,
            self.map,
            self.camera,
            config=self.config,
            feature=feature,
            max_matches=max_matches,
        )
        if not result.success or result.pose_wc is None:
            return result

        frame.set_pose(result.pose_wc)
        self.current_frame = frame
        if insert_keyframe and self.should_insert_keyframe(frame):
            self.insert_keyframe(frame)
        return result


def _empty_local_map_matches() -> LocalMapMatchSet:
    return LocalMapMatchSet(
        point_ids=np.empty(0, dtype=np.int64),
        points_3d=np.empty((0, 3), dtype=np.float64),
        points_2d=np.empty((0, 2), dtype=np.float64),
        frame_keypoint_indices=np.empty(0, dtype=np.int64),
        distances=np.empty(0, dtype=np.float64),
    )


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return points
