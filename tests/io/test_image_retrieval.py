import numpy as np
import pytest

from slam.io.image_retrieval import retrieve_loop_candidates, retrieve_nearest, temporal_exclusion_indices


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


def test_cosine_distance_rejects_zero_descriptors():
    with pytest.raises(ValueError, match="non-zero"):
        retrieve_nearest(np.array([1.0, 0.0]), np.array([[0.0, 0.0]]), metric="cosine")
