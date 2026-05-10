"""Optional learned feature matcher adapters."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np

from slam.features.base import FeatureMatchSet


class OptionalFeatureDependencyError(ImportError):
    """Raised when an optional learned matching dependency is unavailable."""


def _missing_dependency(name: str, extra: str = "modern") -> OptionalFeatureDependencyError:
    return OptionalFeatureDependencyError(
        f"{name} is an optional matcher backend. Install it with `pip install -e .[{extra}]` "
        "and make sure the package supports your Python/platform."
    )


@dataclass(frozen=True)
class LearnedKeypoint:
    """Small keypoint object exposing the OpenCV-compatible `.pt` field."""

    pt: tuple[float, float]


class LightGlueMatcher:
    """Adapter for LightGlue sparse feature matching."""

    name = "lightglue"
    supported_extractors = {
        "aliked": "ALIKED",
        "disk": "DISK",
        "doghardnet": "DoGHardNet",
        "sift": "SIFT",
        "superpoint": "SuperPoint",
    }

    def __init__(
        self,
        *,
        extractor: str = "sift",
        max_features: int = 2048,
        device: str | None = None,
        resize: int | None = None,
    ) -> None:
        try:
            import lightglue as lightglue_module
            import torch
        except ImportError as exc:
            raise _missing_dependency("LightGlue") from exc

        extractor = extractor.lower()
        if extractor not in self.supported_extractors:
            options = ", ".join(sorted(self.supported_extractors))
            raise ValueError(f"LightGlue extractor must be one of: {options}")

        extractor_cls = getattr(lightglue_module, self.supported_extractors[extractor])
        self._torch = torch
        self.extractor_name = extractor
        self.device = _resolve_device(torch, device)
        self.resize = resize
        self.extractor = _module_to_device(
            extractor_cls(max_num_keypoints=max_features),
            self.device,
        )
        self.matcher = _module_to_device(lightglue_module.LightGlue(features=extractor), self.device)

    def match_images(self, image0: np.ndarray, image1: np.ndarray) -> FeatureMatchSet:
        tensor0 = _image_to_torch(image0, self._torch, self.device, channels=3, batch=False)
        tensor1 = _image_to_torch(image1, self._torch, self.device, channels=3, batch=False)

        with _inference_mode(self._torch):
            features0 = _call_lightglue_extract(self.extractor, tensor0, resize=self.resize)
            features1 = _call_lightglue_extract(self.extractor, tensor1, resize=self.resize)
            matches01 = self.matcher({"image0": features0, "image1": features1})

        features0 = _remove_batch_dim(features0)
        features1 = _remove_batch_dim(features1)
        matches01 = _remove_batch_dim(matches01)
        keypoints0 = _keypoints_from_points(_require_array(features0, "keypoints").reshape(-1, 2))
        keypoints1 = _keypoints_from_points(_require_array(features1, "keypoints").reshape(-1, 2))
        match_pairs = _lightglue_match_pairs(matches01)
        distances = _lightglue_distances(matches01, match_pairs)
        matches = _dmatches_from_pairs(match_pairs, distances)
        return FeatureMatchSet(
            keypoints0=keypoints0,
            keypoints1=keypoints1,
            descriptors0=_optional_feature_matrix(features0, "descriptors", len(keypoints0)),
            descriptors1=_optional_feature_matrix(features1, "descriptors", len(keypoints1)),
            matches=matches,
        )


class LoFTRMatcher:
    """Adapter for Kornia LoFTR detector-free matching."""

    name = "loftr"

    def __init__(self, *, pretrained: str = "outdoor", device: str | None = None) -> None:
        try:
            from kornia.feature import LoFTR
            import torch
        except ImportError as exc:
            raise _missing_dependency("Kornia LoFTR") from exc

        self._torch = torch
        self.device = _resolve_device(torch, device)
        self.matcher = _module_to_device(LoFTR(pretrained=pretrained), self.device)

    def match_images(self, image0: np.ndarray, image1: np.ndarray) -> FeatureMatchSet:
        tensor0 = _image_to_torch(image0, self._torch, self.device, channels=1, batch=True)
        tensor1 = _image_to_torch(image1, self._torch, self.device, channels=1, batch=True)

        with _inference_mode(self._torch):
            output = _remove_batch_dim(self.matcher({"image0": tensor0, "image1": tensor1}))

        points0 = _require_array(output, "keypoints0").reshape(-1, 2)
        points1 = _require_array(output, "keypoints1").reshape(-1, 2)
        confidences = _optional_array(output, "confidence")
        distances = np.zeros(len(points0), dtype=np.float32)
        if confidences is not None:
            distances = np.clip(1.0 - confidences.reshape(-1), 0.0, 1.0).astype(np.float32)
        matches = _identity_dmatches(len(points0), distances)
        return FeatureMatchSet(
            keypoints0=_keypoints_from_points(points0),
            keypoints1=_keypoints_from_points(points1),
            descriptors0=None,
            descriptors1=None,
            matches=matches,
        )


def create_learned_matcher(name: str, **kwargs):
    """Create an optional learned matcher by name."""

    normalized = name.lower()
    if normalized == "lightglue":
        return LightGlueMatcher(**kwargs)
    if normalized == "loftr":
        kwargs.pop("max_features", None)
        return LoFTRMatcher(**kwargs)
    raise ValueError("learned matcher must be 'lightglue' or 'loftr'")


def _resolve_device(torch, device: str | None) -> str:
    if device is not None:
        return device
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "is_available") and cuda.is_available():
        return "cuda"
    return "cpu"


def _module_to_device(module, device: str):
    if hasattr(module, "eval"):
        module = module.eval()
    if hasattr(module, "to"):
        return module.to(device)
    if device == "cuda" and hasattr(module, "cuda"):
        return module.cuda()
    return module


def _inference_mode(torch):
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    if hasattr(torch, "no_grad"):
        return torch.no_grad()
    return nullcontext()


def _image_to_torch(
    image: np.ndarray,
    torch,
    device: str,
    *,
    channels: int,
    batch: bool,
):
    array = _normalized_image_array(image, channels=channels)
    if batch:
        array = array[np.newaxis, ...]
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if hasattr(tensor, "to"):
        tensor = tensor.to(device)
    return tensor


def _normalized_image_array(image: np.ndarray, *, channels: int) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        if channels == 3:
            array = np.repeat(array[..., np.newaxis], 3, axis=2)
            array = array.transpose(2, 0, 1)
        else:
            array = array[np.newaxis, :, :]
    elif array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        if channels == 1:
            if array.shape[2] == 1:
                array = array[:, :, 0]
            else:
                array = array.mean(axis=2)
            array = array[np.newaxis, :, :]
        elif channels == 3:
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            if array.shape[2] != 3:
                raise ValueError("image must have 1, 3, or 4 channels")
            array = array.transpose(2, 0, 1)
        else:
            raise ValueError("channels must be 1 or 3")
    else:
        raise ValueError("image must be a 2D grayscale or 3D color array")

    array = array.astype(np.float32, copy=False)
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return array


def _call_lightglue_extract(extractor, image, *, resize: int | None):
    try:
        return extractor.extract(image, resize=resize)
    except TypeError:
        return extractor.extract(image)


def _remove_batch_dim(value):
    if isinstance(value, dict):
        return {key: _remove_batch_dim(item) for key, item in value.items()}
    array = _to_numpy(value)
    if array.ndim > 1 and array.shape[0] == 1:
        return array[0]
    return array


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _require_array(mapping: dict[str, object], key: str) -> np.ndarray:
    if key not in mapping:
        raise RuntimeError(f"learned matcher output is missing '{key}'")
    return _to_numpy(mapping[key]).astype(np.float32, copy=False)


def _optional_array(mapping: dict[str, object], key: str) -> np.ndarray | None:
    if key not in mapping:
        return None
    return _to_numpy(mapping[key]).astype(np.float32, copy=False)


def _optional_feature_matrix(mapping: dict[str, object], key: str, count: int) -> np.ndarray | None:
    values = _optional_array(mapping, key)
    if values is None:
        return None
    values = values.reshape(values.shape[0], -1)
    if values.shape[0] != count and values.shape[1] == count:
        values = values.T
    return values.astype(np.float32, copy=False)


def _lightglue_match_pairs(output: dict[str, object]) -> np.ndarray:
    if "matches" in output:
        return _to_numpy(output["matches"]).astype(np.int64, copy=False).reshape(-1, 2)
    if "matches0" in output:
        matches0 = _to_numpy(output["matches0"]).astype(np.int64, copy=False).reshape(-1)
        query_indices = np.flatnonzero(matches0 >= 0)
        return np.column_stack([query_indices, matches0[query_indices]]).astype(np.int64, copy=False)
    raise RuntimeError("LightGlue output is missing 'matches' or 'matches0'")


def _lightglue_distances(output: dict[str, object], match_pairs: np.ndarray) -> np.ndarray:
    if "scores" in output:
        scores = _to_numpy(output["scores"]).astype(np.float32, copy=False).reshape(-1)
        if len(scores) == len(match_pairs):
            return np.clip(1.0 - scores, 0.0, 1.0)
    if "matching_scores0" in output:
        scores0 = _to_numpy(output["matching_scores0"]).astype(np.float32, copy=False).reshape(-1)
        if len(match_pairs) and np.max(match_pairs[:, 0]) < len(scores0):
            return np.clip(1.0 - scores0[match_pairs[:, 0]], 0.0, 1.0)
    return np.zeros(len(match_pairs), dtype=np.float32)


def _keypoints_from_points(points: np.ndarray) -> list[LearnedKeypoint]:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    return [LearnedKeypoint((float(point[0]), float(point[1]))) for point in points]


def _dmatches_from_pairs(match_pairs: np.ndarray, distances: np.ndarray) -> list[cv2.DMatch]:
    match_pairs = np.asarray(match_pairs, dtype=np.int64).reshape(-1, 2)
    distances = np.asarray(distances, dtype=np.float32).reshape(-1)
    if len(distances) != len(match_pairs):
        raise ValueError("distances must have the same length as match_pairs")
    return [
        cv2.DMatch(_queryIdx=int(query_idx), _trainIdx=int(train_idx), _distance=float(distance))
        for (query_idx, train_idx), distance in zip(match_pairs, distances)
    ]


def _identity_dmatches(count: int, distances: np.ndarray) -> list[cv2.DMatch]:
    pairs = np.column_stack([np.arange(count), np.arange(count)])
    return _dmatches_from_pairs(pairs, distances)
