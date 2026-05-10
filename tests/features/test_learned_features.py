from contextlib import nullcontext
import sys
import types

import numpy as np
import pytest

from slam.features.learned_features import OptionalFeatureDependencyError, create_learned_matcher


def test_unknown_learned_matcher_is_rejected():
    with pytest.raises(ValueError, match="lightglue"):
        create_learned_matcher("unknown")


def test_missing_learned_matcher_dependency_has_install_guidance(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "lightglue":
            raise ImportError("missing lightglue for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalFeatureDependencyError, match=r"pip install -e \.\[modern\]"):
        create_learned_matcher("lightglue")


def test_lightglue_matcher_returns_feature_match_set(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_lightglue(monkeypatch)
    matcher = create_learned_matcher("lightglue", max_features=3, device="cpu")

    result = matcher.match_images(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))

    assert len(result.keypoints0) == 3
    assert len(result.keypoints1) == 3
    assert len(result.matches) == 2
    np.testing.assert_allclose(result.points0, [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(result.points1, [[3.0, 4.0], [1.0, 2.0]])
    np.testing.assert_allclose(result.distances, [0.1, 0.2], atol=1e-6)
    assert result.descriptors0.shape == (3, 2)


def test_loftr_matcher_returns_detector_free_matches(monkeypatch):
    _install_fake_torch(monkeypatch)
    _install_fake_kornia(monkeypatch)
    matcher = create_learned_matcher("loftr", device="cpu")

    result = matcher.match_images(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))

    assert result.descriptors0 is None
    assert result.descriptors1 is None
    assert len(result.matches) == 2
    np.testing.assert_allclose(result.points0, [[1.0, 2.0], [5.0, 6.0]])
    np.testing.assert_allclose(result.points1, [[1.5, 2.5], [5.5, 6.5]])
    np.testing.assert_allclose(result.distances, [0.05, 0.2], atol=1e-6)


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def to(self, _device):
        return self

    def __array__(self, dtype=None):
        return self.array.astype(dtype) if dtype is not None else self.array


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


class _FakeTorch:
    cuda = _FakeCuda()

    @staticmethod
    def from_numpy(array):
        return _FakeTensor(array)

    @staticmethod
    def inference_mode():
        return nullcontext()


def _install_fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)


def _install_fake_lightglue(monkeypatch):
    fake = types.ModuleType("lightglue")

    class SIFT:
        def __init__(self, *, max_num_keypoints):
            self.max_num_keypoints = max_num_keypoints

        def eval(self):
            return self

        def to(self, _device):
            return self

        def extract(self, image, *, resize=None):
            assert np.asarray(image).shape == (3, 8, 8)
            assert resize is None
            return {
                "keypoints": np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
                "descriptors": np.array([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], dtype=np.float32),
            }

    class LightGlue:
        def __init__(self, *, features):
            self.features = features

        def eval(self):
            return self

        def to(self, _device):
            return self

        def __call__(self, _inputs):
            return {
                "matches": np.array([[[0, 1], [2, 0]]], dtype=np.int64),
                "scores": np.array([[0.9, 0.8]], dtype=np.float32),
            }

    fake.SIFT = SIFT
    fake.LightGlue = LightGlue
    monkeypatch.setitem(sys.modules, "lightglue", fake)


def _install_fake_kornia(monkeypatch):
    kornia = types.ModuleType("kornia")
    feature = types.ModuleType("kornia.feature")

    class LoFTR:
        def __init__(self, *, pretrained):
            self.pretrained = pretrained

        def eval(self):
            return self

        def to(self, _device):
            return self

        def __call__(self, inputs):
            assert np.asarray(inputs["image0"]).shape == (1, 1, 8, 8)
            assert np.asarray(inputs["image1"]).shape == (1, 1, 8, 8)
            return {
                "keypoints0": np.array([[1.0, 2.0], [5.0, 6.0]], dtype=np.float32),
                "keypoints1": np.array([[1.5, 2.5], [5.5, 6.5]], dtype=np.float32),
                "confidence": np.array([0.95, 0.8], dtype=np.float32),
            }

    feature.LoFTR = LoFTR
    kornia.feature = feature
    monkeypatch.setitem(sys.modules, "kornia", kornia)
    monkeypatch.setitem(sys.modules, "kornia.feature", feature)
