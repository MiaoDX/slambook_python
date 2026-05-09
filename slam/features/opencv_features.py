"""OpenCV feature detection and descriptor matching."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from slam.features.base import FeatureMatchSet

SUPPORTED_FEATURES = {"orb", "sift"}


def create_detector(feature: str = "orb", *, max_features: int = 1000) -> cv2.Feature2D:
    """Create an OpenCV feature detector by name."""

    feature = feature.lower()
    if feature == "orb":
        return cv2.ORB_create(nfeatures=max_features)
    if feature == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV was built without SIFT; install a build that includes cv2.SIFT_create.")
        return cv2.SIFT_create(nfeatures=max_features)

    options = ", ".join(sorted(SUPPORTED_FEATURES))
    raise ValueError(f"Unknown feature '{feature}'. Supported features: {options}.")


def detect_and_compute(
    image: np.ndarray,
    *,
    feature: str = "orb",
    max_features: int = 1000,
) -> tuple[Sequence[cv2.KeyPoint], np.ndarray | None]:
    """Detect keypoints and compute descriptors for one image."""

    if image is None:
        raise ValueError("image must not be None")

    detector = create_detector(feature, max_features=max_features)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return keypoints, descriptors


def match_descriptors(
    descriptors0: np.ndarray | None,
    descriptors1: np.ndarray | None,
    *,
    feature: str = "orb",
    ratio: float | None = None,
    max_distance_factor: float = 2.0,
    min_distance_floor: float = 30.0,
) -> list[cv2.DMatch]:
    """Match descriptor arrays using defaults suited to ORB or SIFT.

    ORB uses Hamming distance and cross-check matching, then keeps matches no
    farther than `max(max_distance_factor * min_distance, min_distance_floor)`.
    SIFT uses L2 distance with Lowe's ratio test by default.
    """

    if descriptors0 is None or descriptors1 is None:
        return []
    if len(descriptors0) == 0 or len(descriptors1) == 0:
        return []

    feature = feature.lower()
    if feature == "orb":
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = list(matcher.match(descriptors0, descriptors1))
        if not matches:
            return []
        matches.sort(key=lambda match: match.distance)
        max_distance = max(max_distance_factor * matches[0].distance, min_distance_floor)
        return [match for match in matches if match.distance <= max_distance]

    if feature == "sift":
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        ratio = 0.75 if ratio is None else ratio
        candidates = matcher.knnMatch(descriptors0, descriptors1, k=2)
        good_matches: list[cv2.DMatch] = []
        for pair in candidates:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < ratio * second.distance:
                good_matches.append(first)
        good_matches.sort(key=lambda match: match.distance)
        return good_matches

    options = ", ".join(sorted(SUPPORTED_FEATURES))
    raise ValueError(f"Unknown feature '{feature}'. Supported features: {options}.")


def match_images(
    image0: np.ndarray,
    image1: np.ndarray,
    *,
    feature: str = "orb",
    max_features: int = 1000,
    ratio: float | None = None,
) -> FeatureMatchSet:
    """Detect and match features between two images."""

    keypoints0, descriptors0 = detect_and_compute(image0, feature=feature, max_features=max_features)
    keypoints1, descriptors1 = detect_and_compute(image1, feature=feature, max_features=max_features)
    matches = match_descriptors(descriptors0, descriptors1, feature=feature, ratio=ratio)
    return FeatureMatchSet(keypoints0, keypoints1, descriptors0, descriptors1, matches)
