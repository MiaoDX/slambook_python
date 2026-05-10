"""Simple image retrieval helpers for loop-closure examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slam.features.opencv_features import detect_and_compute


class OptionalRetrievalDependencyError(ImportError):
    """Raised when an optional retrieval dependency is unavailable."""


@dataclass(frozen=True)
class RetrievalCandidate:
    """One image retrieval result."""

    index: int
    score: float


@dataclass(frozen=True)
class VisualVocabulary:
    """Bag-of-visual-words cluster centers."""

    centers: np.ndarray

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=np.float64)
        if centers.ndim != 2:
            raise ValueError("centers must be a KxD array")
        if len(centers) == 0:
            raise ValueError("centers must contain at least one visual word")
        object.__setattr__(self, "centers", centers)

    @property
    def word_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def descriptor_dim(self) -> int:
        return int(self.centers.shape[1])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, centers=self.centers)

    @classmethod
    def load(cls, path: str | Path) -> "VisualVocabulary":
        data = np.load(path)
        return cls(data["centers"])


def require_faiss():
    """Import FAISS or raise with project-specific install guidance."""

    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise OptionalRetrievalDependencyError(
            "FAISS is an optional retrieval backend. Install it with `pip install -e .[modern]` "
            "and verify wheel support for your Python/platform."
        ) from exc
    return faiss


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


def retrieve_nearest_faiss(
    query: np.ndarray,
    database: np.ndarray,
    *,
    top_k: int = 5,
    exclude_indices: set[int] | None = None,
) -> list[RetrievalCandidate]:
    """Return nearest descriptor indices using FAISS `IndexFlatL2`."""

    query = np.asarray(query, dtype=np.float32).reshape(-1)
    database = np.asarray(database, dtype=np.float32)
    if database.ndim != 2:
        raise ValueError("database must be an NxD array")
    if database.shape[1] != query.shape[0]:
        raise ValueError("query and database descriptor dimensions must match")
    if top_k <= 0:
        return []
    exclude_indices = exclude_indices or set()

    faiss = require_faiss()
    index = faiss.IndexFlatL2(database.shape[1])
    index.add(np.ascontiguousarray(database))
    distances, indices = index.search(query.reshape(1, -1), len(database))

    candidates = []
    for raw_index, squared_distance in zip(indices[0], distances[0]):
        index_value = int(raw_index)
        if index_value < 0 or index_value in exclude_indices:
            continue
        candidates.append(RetrievalCandidate(index=index_value, score=float(np.sqrt(squared_distance))))
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
    backend: str = "numpy",
) -> list[RetrievalCandidate]:
    """Retrieve loop candidates while excluding immediate temporal neighbors."""

    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be an NxD array")
    exclude = temporal_exclusion_indices(current_index, sequence_length=len(descriptors), window=temporal_window)
    if backend == "faiss":
        if metric != "l2":
            raise ValueError("FAISS backend currently supports only l2 metric")
        return retrieve_nearest_faiss(
            descriptors[current_index],
            descriptors,
            top_k=top_k,
            exclude_indices=exclude,
        )
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'faiss'")
    return retrieve_nearest(
        descriptors[current_index],
        descriptors,
        top_k=top_k,
        metric=metric,
        exclude_indices=exclude,
    )


def mean_pool_descriptors(descriptors: np.ndarray | None, *, output_dim: int) -> np.ndarray:
    """Mean-pool local descriptors into one normalized global descriptor."""

    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(output_dim, dtype=np.float64)

    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be a 2D array")
    pooled = descriptors.mean(axis=0)
    if pooled.shape[0] != output_dim:
        raise ValueError(f"expected descriptor dimension {output_dim}, got {pooled.shape[0]}")
    norm = np.linalg.norm(pooled)
    if norm == 0.0:
        return pooled
    return pooled / norm


def opencv_global_descriptor(image: np.ndarray, *, feature: str = "orb", max_features: int = 1000) -> np.ndarray:
    """Create a simple global descriptor from OpenCV local descriptors."""

    _, descriptors = detect_and_compute(image, feature=feature, max_features=max_features)
    output_dim = 32 if feature.lower() == "orb" else 128
    return mean_pool_descriptors(descriptors, output_dim=output_dim)


def train_visual_vocabulary(
    descriptors: np.ndarray,
    *,
    word_count: int,
    max_iter: int = 50,
    seed: int = 0,
) -> VisualVocabulary:
    """Train a small deterministic k-means visual vocabulary."""

    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be an NxD array")
    if word_count <= 0:
        raise ValueError("word_count must be positive")
    if len(descriptors) < word_count:
        raise ValueError("word_count cannot exceed descriptor count")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    rng = np.random.default_rng(seed)
    centers = descriptors[rng.choice(len(descriptors), size=word_count, replace=False)].copy()
    for _ in range(max_iter):
        labels = _nearest_word_indices(descriptors, centers)
        new_centers = centers.copy()
        for word_index in range(word_count):
            members = descriptors[labels == word_index]
            if len(members):
                new_centers[word_index] = members.mean(axis=0)
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    return VisualVocabulary(centers)


def bow_histogram(
    descriptors: np.ndarray | None,
    vocabulary: VisualVocabulary,
    *,
    normalize: str = "l1",
) -> np.ndarray:
    """Encode local descriptors as a bag-of-visual-words histogram."""

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocabulary.word_count, dtype=np.float64)
    descriptors = np.asarray(descriptors, dtype=np.float64)
    if descriptors.ndim != 2:
        raise ValueError("descriptors must be a 2D array")
    if descriptors.shape[1] != vocabulary.descriptor_dim:
        raise ValueError(
            f"descriptor dimension {descriptors.shape[1]} does not match vocabulary dimension "
            f"{vocabulary.descriptor_dim}"
        )

    labels = _nearest_word_indices(descriptors, vocabulary.centers)
    histogram = np.bincount(labels, minlength=vocabulary.word_count).astype(np.float64)
    if normalize == "none":
        return histogram
    if normalize == "l1":
        norm = histogram.sum()
    elif normalize == "l2":
        norm = np.linalg.norm(histogram)
    else:
        raise ValueError("normalize must be 'none', 'l1', or 'l2'")
    return histogram if norm == 0.0 else histogram / norm


def opencv_bow_descriptor(
    image: np.ndarray,
    vocabulary: VisualVocabulary,
    *,
    feature: str = "orb",
    max_features: int = 1000,
    normalize: str = "l1",
) -> np.ndarray:
    """Create a BoW global descriptor from OpenCV local descriptors."""

    _, descriptors = detect_and_compute(image, feature=feature, max_features=max_features)
    return bow_histogram(descriptors, vocabulary, normalize=normalize)


def _cosine_distance(database: np.ndarray, query: np.ndarray) -> np.ndarray:
    database_norm = np.linalg.norm(database, axis=1)
    query_norm = np.linalg.norm(query)
    if query_norm == 0 or np.any(database_norm == 0):
        raise ValueError("cosine metric requires non-zero descriptors")
    similarity = (database @ query) / (database_norm * query_norm)
    return 1.0 - similarity


def _nearest_word_indices(descriptors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.sum((descriptors[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)
