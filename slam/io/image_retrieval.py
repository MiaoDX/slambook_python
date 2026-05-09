"""Simple image retrieval helpers for loop-closure examples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalCandidate:
    """One image retrieval result."""

    index: int
    score: float


def retrieve_nearest(
    query: np.ndarray,
    database: np.ndarray,
    *,
    top_k: int = 5,
    metric: str = "l2",
    exclude_indices: set[int] | None = None,
) -> list[RetrievalCandidate]:
    """Return nearest database descriptor indices for one query descriptor."""

    query = np.asarray(query, dtype=np.float64).reshape(-1)
    database = np.asarray(database, dtype=np.float64)
    if database.ndim != 2:
        raise ValueError("database must be an NxD array")
    if database.shape[1] != query.shape[0]:
        raise ValueError("query and database descriptor dimensions must match")
    if top_k <= 0:
        return []
    exclude_indices = exclude_indices or set()

    if metric == "l2":
        scores = np.linalg.norm(database - query[None, :], axis=1)
        order = np.argsort(scores)
    elif metric == "cosine":
        scores = _cosine_distance(database, query)
        order = np.argsort(scores)
    else:
        raise ValueError("metric must be 'l2' or 'cosine'")

    candidates = []
    for index in order:
        index = int(index)
        if index in exclude_indices:
            continue
        candidates.append(RetrievalCandidate(index=index, score=float(scores[index])))
        if len(candidates) == top_k:
            break
    return candidates


def temporal_exclusion_indices(current_index: int, *, sequence_length: int, window: int) -> set[int]:
    """Return indices too close in time to be loop-closure candidates."""

    if current_index < 0 or current_index >= sequence_length:
        raise ValueError("current_index must be inside the sequence")
    if window < 0:
        raise ValueError("window must be non-negative")

    start = max(0, current_index - window)
    end = min(sequence_length, current_index + window + 1)
    return set(range(start, end))


def retrieve_loop_candidates(
    descriptors: np.ndarray,
    *,
    current_index: int,
    top_k: int = 5,
    temporal_window: int = 10,
    metric: str = "l2",
) -> list[RetrievalCandidate]:
    """Retrieve loop candidates while excluding immediate temporal neighbors."""

    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be an NxD array")
    exclude = temporal_exclusion_indices(current_index, sequence_length=len(descriptors), window=temporal_window)
    return retrieve_nearest(
        descriptors[current_index],
        descriptors,
        top_k=top_k,
        metric=metric,
        exclude_indices=exclude,
    )


def _cosine_distance(database: np.ndarray, query: np.ndarray) -> np.ndarray:
    database_norm = np.linalg.norm(database, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0 or np.any(database_norm == 0):
        raise ValueError("cosine metric requires non-zero descriptors")
    similarity = (database @ query) / (database_norm * query_norm)
    return 1.0 - similarity
