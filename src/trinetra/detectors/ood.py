"""Reference-based feature-space OOD / distribution-shift detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from trinetra.detectors.near_duplicate import (
    EmbeddingSpaceMismatchError,
    InvalidFeatureBatchError,
    cosine_similarity,
)
from trinetra.features.exceptions import TrinetraFeatureError
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure, ImageId, make_embedding_space_id

OOD_SCHEMA_VERSION = 1
OOD_DETECTOR_NAME = "reference_cosine_distribution_shift"
OOD_DETECTOR_VERSION = "1.0.0"
DEFAULT_NEAREST_COSINE_THRESHOLD = 0.80


class OODDetectionError(ValueError):
    """Base exception for invalid reference/query FeatureBatch inputs."""


@dataclass(frozen=True)
class ReferenceSetMetadata:
    """Identifier and coverage information for the chosen reference batch."""

    identifier: Optional[str]
    identifier_source: Optional[str]
    dataset_name: str
    input_fingerprint: Optional[str]
    embedding_count: int
    failed_image_count: int


@dataclass(frozen=True)
class ReferenceNeighborEvidence:
    """Reference image and exact cosine similarity retained as evidence."""

    image_id: ImageId
    file_path: Optional[Path]
    similarity: float


@dataclass(frozen=True)
class UnscoredQuery:
    """Query image omitted by Layer 1B or lacking a comparable reference vector."""

    image_id: ImageId
    file_path: Optional[Path]
    reason_code: str
    reason: str


@dataclass(frozen=True)
class OODFinding:
    """Analyst-readable evidence that a representation may shift from reference."""

    image_id: ImageId
    file_path: Optional[Path]
    nearest_reference_image_ids: tuple[ImageId, ...]
    nearest_reference_neighbors: tuple[ReferenceNeighborEvidence, ...]
    nearest_reference_similarity: float
    configured_threshold: float
    anomaly_score: float
    embedding_space_id: str
    reference_set_identifier: Optional[str]
    state: str
    interpretation: str
    recommendation: str


@dataclass(frozen=True)
class OODDetectionResult:
    """Typed, versioned output for one reference/query comparison."""

    schema_version: int
    detector_name: str
    detector_version: str
    threshold: float
    reference_set: ReferenceSetMetadata
    embedding_space_id: str
    total_query_images: int
    anomalous_count: int
    findings: tuple[OODFinding, ...]
    unscored_queries: tuple[UnscoredQuery, ...]


def _image_id_key(image_id: ImageId) -> tuple[str, str]:
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise OODDetectionError(f"Unsupported image_id: {image_id!r}")
    return ("int" if isinstance(image_id, int) else "str", str(image_id))


def _validate_batch(batch: FeatureBatch, label: str) -> tuple[str, dict[ImageId, ImageEmbedding]]:
    space_id = batch.embedding_space_id
    if not isinstance(space_id, str) or not space_id.strip():
        raise EmbeddingSpaceMismatchError(f"{label} FeatureBatch has no embedding_space_id")
    try:
        expected = make_embedding_space_id(
            model_id=batch.model_id,
            weights_identity=batch.weights_identity,
            preprocessing=batch.preprocessing,
            embedding_dim=batch.embedding_dim,
            normalization=batch.normalization,
        )
    except (TypeError, ValueError, TrinetraFeatureError) as exc:
        raise EmbeddingSpaceMismatchError(f"Invalid {label} embedding-space metadata: {exc}") from exc
    if expected != space_id:
        raise EmbeddingSpaceMismatchError(f"{label} embedding_space_id does not match its feature metadata")
    if isinstance(batch.embedding_dim, bool) or not isinstance(batch.embedding_dim, int) or batch.embedding_dim <= 0:
        raise InvalidFeatureBatchError(f"{label} embedding_dim must be a positive integer")

    embedding_by_id: dict[ImageId, ImageEmbedding] = {}
    for embedding in batch.embeddings:
        if not isinstance(embedding, ImageEmbedding):
            raise InvalidFeatureBatchError(f"{label} embeddings must contain ImageEmbedding records")
        _image_id_key(embedding.image_id)
        if embedding.image_id in embedding_by_id:
            raise InvalidFeatureBatchError(f"Duplicate {label} image_id: {embedding.image_id!r}")
        record_space = getattr(embedding, "embedding_space_id", space_id)
        if record_space != space_id:
            raise EmbeddingSpaceMismatchError(
                f"{label} image {embedding.image_id!r} belongs to embedding space {record_space!r}, "
                f"expected {space_id!r}"
            )
        vector = np.asarray(embedding.vector)
        if vector.ndim != 1 or vector.size != batch.embedding_dim:
            raise InvalidFeatureBatchError(
                f"{label} image {embedding.image_id!r} has dimension {vector.size}; "
                f"expected {batch.embedding_dim}"
            )
        if not (np.issubdtype(vector.dtype, np.integer) or np.issubdtype(vector.dtype, np.floating)):
            raise InvalidFeatureBatchError(f"{label} image {embedding.image_id!r} has non-real numeric values")
        if not np.isfinite(vector).all():
            raise InvalidFeatureBatchError(f"{label} image {embedding.image_id!r} has NaN or infinite values")
        if embedding.file_path is not None and not isinstance(embedding.file_path, Path):
            raise InvalidFeatureBatchError(f"{label} image {embedding.image_id!r} has invalid file_path")
        embedding_by_id[embedding.image_id] = embedding

    failure_ids: set[ImageId] = set()
    for failure in batch.failures:
        if not isinstance(failure, ImageFailure):
            raise InvalidFeatureBatchError(f"{label} failures must contain ImageFailure records")
        _image_id_key(failure.image_id)
        if failure.image_id in embedding_by_id or failure.image_id in failure_ids:
            raise InvalidFeatureBatchError(f"Duplicate or overlapping {label} image_id: {failure.image_id!r}")
        failure_ids.add(failure.image_id)
    return space_id, embedding_by_id


class OODDetector:
    """Compare query features with their nearest exact-cosine reference vector.

    Search uses O(query × reference) time and O(reference + query evidence)
    memory, without constructing a combined pairwise similarity matrix.
    """

    def __init__(self, threshold: float = DEFAULT_NEAREST_COSINE_THRESHOLD) -> None:
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not -1.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("threshold must be finite and in [-1, 1]")
        self.threshold = float(threshold)

    def detect(
        self,
        reference_batch: FeatureBatch,
        query_batch: FeatureBatch,
        *,
        reference_set_id: Optional[str] = None,
    ) -> OODDetectionResult:
        """Detect feature-space distance from a reference batch.

        Queries whose IDs also occur in the reference batch exclude those exact
        IDs from their neighbor candidates. If no comparable reference remains,
        the query is marked unscored rather than anomalous.
        """
        if not isinstance(reference_batch, FeatureBatch) or not isinstance(query_batch, FeatureBatch):
            raise TypeError("OODDetector consumes reference and query FeatureBatch objects")
        reference_space, references = _validate_batch(reference_batch, "reference")
        query_space, queries = _validate_batch(query_batch, "query")
        if reference_batch.embedding_dim != query_batch.embedding_dim:
            raise InvalidFeatureBatchError(
                f"Reference/query embedding dimensions differ: "
                f"{reference_batch.embedding_dim} != {query_batch.embedding_dim}"
            )
        if reference_space != query_space:
            raise EmbeddingSpaceMismatchError(
                "Reference and query FeatureBatch objects use different embedding spaces"
            )
        if reference_set_id is not None and (not isinstance(reference_set_id, str) or not reference_set_id.strip()):
            raise ValueError("reference_set_id must be a non-empty string when provided")
        chosen_reference_id = reference_set_id or reference_batch.dataset_name
        chosen_reference_source = "caller" if reference_set_id is not None else "feature_batch.dataset_name"
        reference_meta = ReferenceSetMetadata(
            identifier=chosen_reference_id,
            identifier_source=chosen_reference_source,
            dataset_name=reference_batch.dataset_name,
            input_fingerprint=reference_batch.input_fingerprint,
            embedding_count=len(references),
            failed_image_count=len(reference_batch.failures),
        )

        findings: list[OODFinding] = []
        unscored: list[UnscoredQuery] = []
        for failure in query_batch.failures:
            unscored.append(UnscoredQuery(
                failure.image_id, failure.file_path, failure.failure_code,
                f"Layer 1B did not produce an embedding: {failure.reason}",
            ))
        for query_id in sorted(queries, key=_image_id_key):
            query = queries[query_id]
            candidates = [
                reference for reference_id, reference in references.items()
                if reference_id != query_id
            ]
            if not candidates:
                empty_reference = not references
                unscored.append(UnscoredQuery(
                    query_id,
                    query.file_path,
                    "empty_reference_set" if empty_reference else "no_comparable_reference",
                    (
                        "Reference FeatureBatch has no successful embeddings, so this query was not scored."
                        if empty_reference else
                        "No reference embedding remains after excluding the same image ID, so this query was not scored."
                    ),
                ))
                continue
            scored = [
                (cosine_similarity(query.vector, reference.vector), reference)
                for reference in candidates
            ]
            best_similarity = max(similarity for similarity, _ in scored)
            best = min(
                ((similarity, reference) for similarity, reference in scored if similarity == best_similarity),
                key=lambda item: _image_id_key(item[1].image_id),
            )
            neighbors = tuple(
                ReferenceNeighborEvidence(reference.image_id, reference.file_path, similarity)
                for similarity, reference in (best,)
            )
            score = max(0.0, self.threshold - best_similarity)
            is_shifted = best_similarity < self.threshold
            findings.append(OODFinding(
                image_id=query_id,
                file_path=query.file_path,
                nearest_reference_image_ids=tuple(neighbor.image_id for neighbor in neighbors),
                nearest_reference_neighbors=neighbors,
                nearest_reference_similarity=best_similarity,
                configured_threshold=self.threshold,
                anomaly_score=score,
                embedding_space_id=query_space,
                reference_set_identifier=chosen_reference_id,
                state="distribution_shift_candidate" if is_shifted else "in_distribution",
                interpretation=(
                    "Feature representation is distant from the reference distribution."
                    if is_shifted else
                    "Feature representation is within the configured nearest-reference similarity threshold."
                ),
                recommendation="REVIEW" if is_shifted else "NO_AUTOMATIC_ACTION",
            ))

        findings.sort(key=lambda finding: _image_id_key(finding.image_id))
        unscored.sort(key=lambda item: _image_id_key(item.image_id))
        total_query_images = len(queries) + len(query_batch.failures)
        return OODDetectionResult(
            schema_version=OOD_SCHEMA_VERSION,
            detector_name=OOD_DETECTOR_NAME,
            detector_version=OOD_DETECTOR_VERSION,
            threshold=self.threshold,
            reference_set=reference_meta,
            embedding_space_id=query_space,
            total_query_images=total_query_images,
            anomalous_count=sum(finding.state == "distribution_shift_candidate" for finding in findings),
            findings=tuple(findings),
            unscored_queries=tuple(unscored),
        )


__all__ = [
    "DEFAULT_NEAREST_COSINE_THRESHOLD",
    "OODDetectionError",
    "OODDetectionResult",
    "OODDetector",
    "OODFinding",
    "OOD_DETECTOR_NAME",
    "OOD_DETECTOR_VERSION",
    "OOD_SCHEMA_VERSION",
    "ReferenceNeighborEvidence",
    "ReferenceSetMetadata",
    "UnscoredQuery",
]
