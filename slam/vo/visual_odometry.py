"""Data model for the Chapter 9 mini visual odometry project."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from slam.camera.pinhole import CameraIntrinsics


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


def _points2(points: np.ndarray, *, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must be an Nx2 array")
    return points
