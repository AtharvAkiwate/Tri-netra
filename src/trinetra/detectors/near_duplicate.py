"""Exact, deterministic near-duplicate detection over Layer 1B embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from trinetra.features.exceptions import TrinetraFeatureError
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageId, make_embedding_space_id

NEAR_DUPLICATE_SCHEMA_VERSION = 1
DEFAULT_SIMILARITY_THRESHOLD = 0.95


class NearDuplicateError(ValueError):
    """Base exception for invalid near-duplicate input or configuration."""


class EmbeddingSpaceMismatchError(NearDuplicateError):
    """Raised when vectors do not all belong to the same embedding space."""


class InvalidFeatureBatchError(NearDuplicateError):
    """Raised when a FeatureBatch is inconsistent or contains unsafe vectors."""


@dataclass(frozen=True)
class DuplicatePair:
    """Analyst-readable evidence for one qualifying unordered image pair."""

    image_a_id: ImageId
    image_a_path: Optional[Path]
    image_b_id: ImageId
    image_b_path: Optional[Path]
    similarity: float
    threshold: float
    embedding_space_id: str


@dataclass(frozen=True)
class DuplicateMember:
    """Image identity and source path retained inside a duplicate cluster."""

    image_id: ImageId
    file_path: Optional[Path]
    embedding_space_id: str


@dataclass(frozen=True)
class DuplicateCluster:
    """Connected component of candidate duplicate relationships."""

    cluster_id: int
    members: tuple[DuplicateMember, ...]
    pairs: tuple[DuplicatePair, ...]


@dataclass(frozen=True)
class NearDuplicateResult:
    """Versioned result and evidence from a Layer 1C detection run."""

    schema_version: int
    threshold: float
    embedding_space_id: str
    total_images: int
    candidate_pair_count: int
    duplicate_cluster_count: int
    pairs: tuple[DuplicatePair, ...]
    clusters: tuple[DuplicateCluster, ...]


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Compute safe cosine similarity; zero vectors have similarity 0.0.

    Inputs are promoted to float64 for norm/dot calculation. Non-finite,
    non-vector, empty, or dimension-mismatched inputs are rejected explicitly.
    """
    a = np.asarray(vector_a)
    b = np.asarray(vector_b)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or b.size == 0:
        raise InvalidFeatureBatchError("Cosine similarity requires non-empty 1-D vectors")
    if a.shape != b.shape:
        raise InvalidFeatureBatchError(
            f"Embedding dimension mismatch: {a.size} != {b.size}"
        )
    if not _is_real_numeric(a.dtype) or not _is_real_numeric(b.dtype):
        raise InvalidFeatureBatchError("Embedding vectors must contain real numeric values")
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    if not np.isfinite(a64).all() or not np.isfinite(b64).all():
        raise InvalidFeatureBatchError("Embedding vectors must contain only finite values")
    # Scale before taking norms to avoid overflow for unusually large values.
    scale_a = float(np.max(np.abs(a64)))
    scale_b = float(np.max(np.abs(b64)))
    if scale_a == 0.0 or scale_b == 0.0:
        return 0.0
    a_scaled, b_scaled = a64 / scale_a, b64 / scale_b
    norm_a, norm_b = float(np.linalg.norm(a_scaled)), float(np.linalg.norm(b_scaled))
    if not math.isfinite(norm_a) or not math.isfinite(norm_b) or norm_a == 0.0 or norm_b == 0.0:
        raise InvalidFeatureBatchError("Embedding vector norm is invalid")
    similarity = float(np.dot(a_scaled, b_scaled) / (norm_a * norm_b))
    if not math.isfinite(similarity):
        raise InvalidFeatureBatchError("Cosine similarity calculation was non-finite")
    return float(np.clip(similarity, -1.0, 1.0))


def _is_real_numeric(dtype: np.dtype) -> bool:
    return np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)


def _id_order(image_id: ImageId) -> tuple[str, str]:
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise InvalidFeatureBatchError(f"Unsupported image_id: {image_id!r}")
    return ("int" if isinstance(image_id, int) else "str", str(image_id))


class NearDuplicateDetector:
    """Find exact cosine-similarity pairs and connected duplicate clusters.

    Pair evaluation is O(n²) time and O(n + e) working/result memory (rather
    than allocating an n-by-n similarity matrix), where e is reported pairs.
    """

    def __init__(self, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a finite number in [-1, 1]")
        if not math.isfinite(float(threshold)) or not -1.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be a finite number in [-1, 1]")
        self.threshold = float(threshold)

    def detect(self, batch: FeatureBatch) -> NearDuplicateResult:
        """Detect candidates from successful embeddings in a Layer 1B batch."""
        if not isinstance(batch, FeatureBatch):
            raise TypeError("NearDuplicateDetector consumes a Layer 1B FeatureBatch")
        space_id = batch.embedding_space_id
        if not isinstance(space_id, str) or not space_id.strip():
            raise EmbeddingSpaceMismatchError("FeatureBatch has no embedding_space_id")
        try:
            expected_space = make_embedding_space_id(
                model_id=batch.model_id,
                weights_identity=batch.weights_identity,
                preprocessing=batch.preprocessing,
                embedding_dim=batch.embedding_dim,
                normalization=batch.normalization,
            )
        except (TypeError, ValueError, TrinetraFeatureError) as exc:
            raise EmbeddingSpaceMismatchError(f"Invalid embedding-space metadata: {exc}") from exc
        if space_id != expected_space:
            raise EmbeddingSpaceMismatchError(
                "FeatureBatch embedding_space_id does not match its model/preprocessing metadata"
            )
        if isinstance(batch.embedding_dim, bool) or not isinstance(batch.embedding_dim, int) or batch.embedding_dim <= 0:
            raise InvalidFeatureBatchError("FeatureBatch embedding_dim must be a positive integer")

        records = list(batch.embeddings)
        seen_ids: set[ImageId] = set()
        vectors: list[np.ndarray] = []
        for record in records:
            if not isinstance(record, ImageEmbedding):
                raise InvalidFeatureBatchError("FeatureBatch embeddings must contain ImageEmbedding records")
            image_id = record.image_id
            _id_order(image_id)
            if image_id in seen_ids:
                raise InvalidFeatureBatchError(f"Duplicate image_id in FeatureBatch: {image_id!r}")
            seen_ids.add(image_id)
            record_space = getattr(record, "embedding_space_id", space_id)
            if record_space != space_id:
                raise EmbeddingSpaceMismatchError(
                    f"Image {image_id!r} belongs to embedding space {record_space!r}, "
                    f"expected {space_id!r}"
                )
            vector = np.asarray(record.vector)
            if vector.ndim != 1 or vector.size != batch.embedding_dim:
                raise InvalidFeatureBatchError(
                    f"Image {image_id!r} has embedding dimension {vector.size}; "
                    f"expected {batch.embedding_dim}"
                )
            if not _is_real_numeric(vector.dtype) or not np.isfinite(vector).all():
                raise InvalidFeatureBatchError(f"Image {image_id!r} has non-finite or non-real-numeric embedding values")
            if record.file_path is not None and not isinstance(record.file_path, Path):
                raise InvalidFeatureBatchError(f"Image {image_id!r} has an invalid file_path")
            vectors.append(vector)

        ordered = sorted(zip(records, vectors, strict=True), key=lambda pair: _id_order(pair[0].image_id))
        ordered_records = [item[0] for item in ordered]
        ordered_vectors = [item[1] for item in ordered]
        pairs: list[DuplicatePair] = []
        for index, (record_a, vector_a) in enumerate(zip(ordered_records, ordered_vectors, strict=True)):
            for other_index in range(index + 1, len(ordered_records)):
                record_b, vector_b = ordered_records[other_index], ordered_vectors[other_index]
                similarity = cosine_similarity(vector_a, vector_b)
                if similarity >= self.threshold:
                    pairs.append(DuplicatePair(
                        image_a_id=record_a.image_id,
                        image_a_path=record_a.file_path,
                        image_b_id=record_b.image_id,
                        image_b_path=record_b.file_path,
                        similarity=similarity,
                        threshold=self.threshold,
                        embedding_space_id=space_id,
                    ))
        clusters = self._clusters(ordered_records, pairs, space_id)
        return NearDuplicateResult(
            schema_version=NEAR_DUPLICATE_SCHEMA_VERSION,
            threshold=self.threshold,
            embedding_space_id=space_id,
            total_images=len(records),
            candidate_pair_count=len(pairs),
            duplicate_cluster_count=len(clusters),
            pairs=tuple(pairs),
            clusters=tuple(clusters),
        )

    @staticmethod
    def _clusters(
        records: list[object], pairs: list[DuplicatePair], space_id: str,
    ) -> list[DuplicateCluster]:
        if not pairs:
            return []
        ids = [record.image_id for record in records]
        parent = list(range(len(ids)))
        index_by_id = {image_id: index for index, image_id in enumerate(ids)}

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for pair in pairs:
            left, right = find(index_by_id[pair.image_a_id]), find(index_by_id[pair.image_b_id])
            if left != right:
                parent[max(left, right)] = min(left, right)
        grouped: dict[int, list[int]] = {}
        for index in range(len(ids)):
            root = find(index)
            grouped.setdefault(root, []).append(index)
        edge_groups: dict[int, list[DuplicatePair]] = {root: [] for root in grouped}
        for pair in pairs:
            edge_groups[find(index_by_id[pair.image_a_id])].append(pair)
        clusters: list[DuplicateCluster] = []
        # ``records`` is sorted by stable ID, so component and member order is fixed.
        for root, member_indices in sorted(grouped.items(), key=lambda entry: entry[1][0]):
            component_pairs = tuple(edge_groups[root])
            if not component_pairs:
                continue
            members = tuple(
                DuplicateMember(ids[index], records[index].file_path, space_id)
                for index in member_indices
            )
            clusters.append(DuplicateCluster(len(clusters) + 1, members, component_pairs))
        return clusters


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DuplicateCluster",
    "DuplicateMember",
    "DuplicatePair",
    "EmbeddingSpaceMismatchError",
    "InvalidFeatureBatchError",
    "NEAR_DUPLICATE_SCHEMA_VERSION",
    "NearDuplicateDetector",
    "NearDuplicateError",
    "NearDuplicateResult",
    "cosine_similarity",
]
