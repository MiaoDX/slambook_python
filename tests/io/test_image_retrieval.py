import builtins
import sys
import types

import numpy as np
import pytest

from slam.io.image_retrieval import (
    OptionalRetrievalDependencyError,
    VisualVocabulary,
    bow_histogram,
    mean_pool_descriptors,
    opencv_bow_descriptor,
    require_faiss,
    retrieve_loop_candidates,
    retrieve_nearest,
    retrieve_nearest_faiss,
    temporal_exclusion_indices,
    train_visual_vocabulary,
)


def test_retrieve_nearest_returns_expected_l2_order():
    database = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [10.0, 10.0]])
    query = np.array([0.9, 1.1])

    candidates = retrieve_nearest(query, database, top_k=3)

    assert [candidate.index for candidate in candidates] == [2, 0, 1]
    assert candidates[0].score < candidates[1].score


def test_retrieve_nearest_supports_cosine_distance():
    database = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    query = np.array([1.0, 0.0])

    candidates = retrieve_nearest(query, database, top_k=2, metric="cosine")

    assert [candidate.index for candidate in candidates] == [0, 2]


def test_temporal_exclusion_indices_clamps_to_sequence_bounds():
    assert temporal_exclusion_indices(1, sequence_length=5, window=2) == {0, 1, 2, 3}
    assert temporal_exclusion_indices(4, sequence_length=5, window=2) == {2, 3, 4}


def test_retrieve_loop_candidates_excludes_temporal_neighbors():
    descriptors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [9.0, 9.0],
            [0.05, 0.02],
        ]
    )

    candidates = retrieve_loop_candidates(descriptors, current_index=1, top_k=2, temporal_window=1)

    assert [candidate.index for candidate in candidates] == [4, 3]


def test_retrieve_loop_candidates_supports_faiss_backend(monkeypatch):
    _install_fake_faiss(monkeypatch)
    descriptors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [9.0, 9.0],
            [0.05, 0.02],
        ]
    )

    candidates = retrieve_loop_candidates(
        descriptors,
        current_index=1,
        top_k=2,
        temporal_window=1,
        backend="faiss",
    )

    assert [candidate.index for candidate in candidates] == [4, 3]


def test_retrieve_nearest_faiss_matches_l2_order_and_scores(monkeypatch):
    _install_fake_faiss(monkeypatch)
    database = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0], [10.0, 10.0]])
    query = np.array([0.9, 1.1])

    candidates = retrieve_nearest_faiss(query, database, top_k=3)

    assert [candidate.index for candidate in candidates] == [2, 0, 1]
    np.testing.assert_allclose(candidates[0].score, np.linalg.norm(database[2] - query), atol=1e-6)


def test_require_faiss_has_modern_install_guidance(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faiss":
            raise ImportError("missing faiss for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OptionalRetrievalDependencyError, match=r"pip install -e \.\[modern\]"):
        require_faiss()


def test_faiss_backend_rejects_cosine_metric(monkeypatch):
    _install_fake_faiss(monkeypatch)
    with pytest.raises(ValueError, match="only l2"):
        retrieve_loop_candidates(np.ones((3, 2)), current_index=0, metric="cosine", backend="faiss")


def test_cosine_distance_rejects_zero_descriptors():
    with pytest.raises(ValueError, match="non-zero"):
        retrieve_nearest(np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), metric="cosine")


def test_mean_pool_descriptors_normalizes_local_descriptor_average():
    descriptors = np.array([[3.0, 0.0], [0.0, 4.0]])

    pooled = mean_pool_descriptors(descriptors, output_dim=2)

    np.testing.assert_allclose(np.linalg.norm(pooled), 1.0)
    np.testing.assert_allclose(pooled, np.array([1.5, 2.0]) / 2.5)


def test_mean_pool_descriptors_returns_zero_for_missing_descriptors():
    np.testing.assert_allclose(mean_pool_descriptors(None, output_dim=3), np.zeros(3))


def test_train_visual_vocabulary_and_bow_histogram_cluster_descriptors(tmp_path):
    descriptors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )

    vocabulary = train_visual_vocabulary(descriptors, word_count=2, seed=1)
    histogram = bow_histogram(descriptors[:3], vocabulary)
    path = tmp_path / "vocabulary.npz"
    vocabulary.save(path)
    loaded = VisualVocabulary.load(path)

    assert vocabulary.centers.shape == (2, 2)
    np.testing.assert_allclose(loaded.centers, vocabulary.centers)
    np.testing.assert_allclose(histogram.sum(), 1.0)
    assert np.count_nonzero(histogram) == 2


def test_bow_histogram_rejects_descriptor_dimension_mismatch():
    vocabulary = VisualVocabulary(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="dimension"):
        bow_histogram(np.zeros((4, 2)), vocabulary)


def test_opencv_bow_descriptor_returns_zero_for_textureless_image():
    vocabulary = VisualVocabulary(np.zeros((3, 32)))
    image = np.zeros((32, 32), dtype=np.uint8)

    descriptor = opencv_bow_descriptor(image, vocabulary, feature="orb")

    np.testing.assert_allclose(descriptor, np.zeros(3))


def _install_fake_faiss(monkeypatch):
    class FakeIndexFlatL2:
        def __init__(self, dimension):
            self.dimension = dimension
            self.database = None

        def add(self, database):
            database = np.asarray(database, dtype=np.float32)
            assert database.shape[1] == self.dimension
            self.database = database

        def search(self, queries, k):
            queries = np.asarray(queries, dtype=np.float32)
            distances = np.sum((self.database[None, :, :] - queries[:, None, :]) ** 2, axis=2)
            order = np.argsort(distances, axis=1)[:, :k]
            sorted_distances = np.take_along_axis(distances, order, axis=1)
            return sorted_distances.astype(np.float32), order.astype(np.int64)

    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(IndexFlatL2=FakeIndexFlatL2))
