"""Neighborhood-based label consistency analysis (Layer 1D)."""

from __future__ import annotations

import math
from collections import Counter
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
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageId, make_embedding_space_id
from trinetra.parsers.models import Dataset, DatasetImage

LABEL_MANIPULATION_SCHEMA_VERSION = 1
LABEL_MANIPULATION_DETECTOR_NAME = "label_manipulation_neighborhood"
LABEL_MANIPULATION_DETECTOR_VERSION = "1.0.0"
DEFAULT_NEIGHBOR_COUNT = 5
DEFAULT_DISAGREEMENT_THRESHOLD = 0.8
DEFAULT_MIN_SAME_CLASS_NEIGHBORS = 1
MetadataValue = Union[str, int]


class LabelManipulationError(ValueError):
    """Base exception for invalid Layer 1D inputs or configuration."""


class MissingEmbeddingError(LabelManipulationError):
    """A labeled dataset image has no corresponding Layer 1B embedding."""


class DatasetFeatureMismatchError(LabelManipulationError):
    """Dataset and FeatureBatch image identities do not align."""


@dataclass(frozen=True)
class LabelReference:
    """A dataset category identity, retaining its human-readable name if known."""

    category_id: int
    category_name: Optional[str]


@dataclass(frozen=True)
class ContributorSourceMetadata:
    """Optional source/contributor/batch values found in caller-supplied metadata."""

    contributor: Optional[MetadataValue] = None
    source: Optional[MetadataValue] = None
    batch: Optional[MetadataValue] = None


@dataclass(frozen=True)
class NeighborEvidence:
    """One exact-cosine neighbor and its original class assignments."""

    image_id: ImageId
    file_path: Optional[Path]
    labels: tuple[LabelReference, ...]
    similarity: float


@dataclass(frozen=True)
class NeighborClassCount:
    """Count of top-k neighbor images carrying one category label."""

    label: LabelReference
    count: int


@dataclass(frozen=True)
class LabelManipulationConfiguration:
    """Persisted operating configuration used to interpret findings."""

    k: int
    disagreement_threshold: float
    minimum_same_class_neighbors: int
    score_name: str = "label_inconsistency_score"
    score_definition: str = "different_class_neighbor_count / available_neighbor_count"


@dataclass(frozen=True)
class LabelManipulationFinding:
    """Per-image evidence; REVIEW is an analyst signal, not a poisoning claim."""

    image_id: ImageId
    file_path: Optional[Path]
    assigned_labels: tuple[LabelReference, ...]
    nearest_neighbors: tuple[NeighborEvidence, ...]
    nearest_neighbor_class_distribution: tuple[NeighborClassCount, ...]
    same_class_neighbor_count: int
    different_class_neighbor_count: int
    strongest_competing_label: Optional[LabelReference]
    similarity_to_same_class_neighbors: Optional[float]
    similarity_to_strongest_competing_class: Optional[float]
    label_inconsistency_score: float
    configured_threshold: float
    embedding_space_id: str
    state: str
    recommendation: str
    state_reason: str
    source_metadata: ContributorSourceMetadata


@dataclass(frozen=True)
class ExcludedImage:
    """Image excluded from analysis with a concrete, non-fabricated reason."""

    image_id: ImageId
    file_path: Optional[Path]
    exclusion_code: str
    reason: str
    source_metadata: ContributorSourceMetadata


@dataclass(frozen=True)
class LabelManipulationResult:
    """Typed and versioned Layer 1D analysis result."""

    schema_version: int
    detector_name: str
    detector_version: str
    configuration: LabelManipulationConfiguration
    embedding_space_id: str
    total_images_analyzed: int
    suspicious_count: int
    findings: tuple[LabelManipulationFinding, ...]
    excluded_images: tuple[ExcludedImage, ...]
    dataset_source_metadata: ContributorSourceMetadata


def _id_key(image_id: ImageId) -> tuple[str, str]:
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise DatasetFeatureMismatchError(f"Unsupported image_id: {image_id!r}")
    return ("int" if isinstance(image_id, int) else "str", str(image_id))


def _metadata_value(value: object) -> Optional[MetadataValue]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        return value if not isinstance(value, str) or value.strip() else None
    return None


def _source_metadata(dataset: Dataset, image: Optional[DatasetImage] = None) -> ContributorSourceMetadata:
    """Read only explicitly present optional metadata; never infer identity."""
    values: dict[str, Optional[MetadataValue]] = {"contributor": None, "source": None, "batch": None}
    aliases = {
        "contributor": ("contributor", "contributor_id"),
        "source": ("source", "source_id"),
        "batch": ("batch", "batch_id"),
    }
    for owner in (dataset, image):
        if owner is None:
            continue
        metadata = getattr(owner, "metadata", None)
        for field_name, field_aliases in aliases.items():
            for alias in field_aliases:
                candidate = getattr(owner, alias, None)
                if candidate is None and isinstance(metadata, dict):
                    candidate = metadata.get(alias)
                normalized = _metadata_value(candidate)
                if normalized is not None:
                    values[field_name] = normalized
                    break
    return ContributorSourceMetadata(**values)


class LabelManipulationDetector:
    """Flag label-neighborhood disagreement for analyst review.

    Exact top-k cosine search takes O(n²) time while retaining no NxN matrix.
    The default score threshold and minimum same-class support are configurable
    development operating values, not validated poisoning criteria.
    """

    def __init__(
        self,
        k: int = DEFAULT_NEIGHBOR_COUNT,
        disagreement_threshold: float = DEFAULT_DISAGREEMENT_THRESHOLD,
        minimum_same_class_neighbors: int = DEFAULT_MIN_SAME_CLASS_NEIGHBORS,
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if (
            isinstance(disagreement_threshold, bool)
            or not isinstance(disagreement_threshold, (int, float))
            or not math.isfinite(float(disagreement_threshold))
            or not 0.0 <= float(disagreement_threshold) <= 1.0
        ):
            raise ValueError("disagreement_threshold must be finite and in [0, 1]")
        if (
            isinstance(minimum_same_class_neighbors, bool)
            or not isinstance(minimum_same_class_neighbors, int)
            or minimum_same_class_neighbors < 0
        ):
            raise ValueError("minimum_same_class_neighbors must be a non-negative integer")
        self.k = k
        self.disagreement_threshold = float(disagreement_threshold)
        self.minimum_same_class_neighbors = minimum_same_class_neighbors

    def detect(self, dataset: Dataset, feature_batch: FeatureBatch) -> LabelManipulationResult:
        """Analyze labeled Dataset images using already computed embeddings only."""
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Layer 1A Dataset")
        if not isinstance(feature_batch, FeatureBatch):
            raise TypeError("feature_batch must be a Layer 1B FeatureBatch")
        space_id = self._validate_feature_batch(dataset, feature_batch)
        source_metadata = _source_metadata(dataset)
        images_by_id: dict[ImageId, DatasetImage] = {}
        image_labels: dict[ImageId, tuple[LabelReference, ...]] = {}
        excluded: list[ExcludedImage] = []

        for image in dataset.images:
            _id_key(image.image_id)
            if image.image_id in images_by_id:
                raise DatasetFeatureMismatchError(f"Duplicate dataset image_id: {image.image_id!r}")
            images_by_id[image.image_id] = image
            raw_labels: dict[int, set[str]] = {}
            for annotation in image.annotations:
                category_id = getattr(annotation, "category_id", None)
                if isinstance(category_id, bool) or not isinstance(category_id, int):
                    continue
                name = getattr(annotation, "category_name", None)
                if not isinstance(name, str) or not name.strip():
                    name = dataset.categories.get(category_id)
                raw_labels.setdefault(category_id, set())
                if isinstance(name, str) and name.strip():
                    raw_labels[category_id].add(name.strip())
                if isinstance(dataset.categories.get(category_id), str) and dataset.categories[category_id].strip():
                    raw_labels[category_id].add(dataset.categories[category_id].strip())
            labels = tuple(
                LabelReference(category_id, self._category_name(dataset, category_id, names))
                for category_id, names in sorted(raw_labels.items())
            )
            image_labels[image.image_id] = labels
            if not image.annotations:
                excluded.append(ExcludedImage(
                    image.image_id, image.file_path, "missing_annotation",
                    "Image has no category annotations and cannot be checked for label consistency.",
                    _source_metadata(dataset, image),
                ))
            elif not labels:
                excluded.append(ExcludedImage(
                    image.image_id, image.file_path, "missing_category",
                    "Image annotations do not contain a usable category ID.",
                    _source_metadata(dataset, image),
                ))

        vectors_by_id = {item.image_id: item for item in feature_batch.embeddings}
        analyzed_ids = [image_id for image_id, labels in image_labels.items() if labels]
        for image_id in analyzed_ids:
            if image_id not in vectors_by_id:
                image = images_by_id[image_id]
                raise MissingEmbeddingError(
                    f"Labeled image {image_id!r} at {image.file_path} has no matching Layer 1B embedding"
                )
        eligible_ids = sorted(analyzed_ids, key=_id_key)
        findings: list[LabelManipulationFinding] = []
        for image_id in eligible_ids:
            image = images_by_id[image_id]
            assigned = image_labels[image_id]
            assigned_ids = {label.category_id for label in assigned}
            target_vector = vectors_by_id[image_id].vector
            candidates: list[NeighborEvidence] = []
            for other_id in eligible_ids:
                if other_id == image_id:
                    continue
                other_image = images_by_id[other_id]
                similarity = cosine_similarity(target_vector, vectors_by_id[other_id].vector)
                candidates.append(NeighborEvidence(
                    other_id,
                    other_image.file_path,
                    image_labels[other_id],
                    similarity,
                ))
            candidates.sort(key=lambda neighbor: (-neighbor.similarity, _id_key(neighbor.image_id)))
            nearest = tuple(candidates[: self.k])
            same_neighbors = [
                neighbor for neighbor in nearest
                if assigned_ids.intersection(label.category_id for label in neighbor.labels)
            ]
            different_neighbors = [neighbor for neighbor in nearest if neighbor not in same_neighbors]
            distribution_counts: Counter[int] = Counter(
                label.category_id for neighbor in nearest for label in neighbor.labels
            )
            label_by_id = {
                label.category_id: label
                for neighbor in nearest for label in neighbor.labels
            }
            distribution = tuple(
                NeighborClassCount(label_by_id[category_id], count)
                for category_id, count in sorted(
                    distribution_counts.items(),
                    key=lambda entry: (-entry[1], _id_key(entry[0])),
                )
            )
            competitor_counts: Counter[int] = Counter(
                label.category_id
                for neighbor in different_neighbors
                for label in neighbor.labels
                if label.category_id not in assigned_ids
            )
            competitor_similarity: dict[int, list[float]] = {}
            for neighbor in different_neighbors:
                for label in neighbor.labels:
                    if label.category_id not in assigned_ids:
                        competitor_similarity.setdefault(label.category_id, []).append(neighbor.similarity)
            strongest_id = None
            if competitor_counts:
                strongest_id = min(
                    competitor_counts,
                    key=lambda category_id: (
                        -competitor_counts[category_id],
                        -float(np.mean(competitor_similarity[category_id])),
                        _id_key(category_id),
                    ),
                )
            strongest = label_by_id.get(strongest_id) if strongest_id is not None else None
            same_similarity = float(np.mean([n.similarity for n in same_neighbors])) if same_neighbors else None
            strongest_similarity = (
                float(np.mean(competitor_similarity[strongest_id])) if strongest_id is not None else None
            )
            different_count = len(different_neighbors)
            score = different_count / len(nearest) if nearest else 0.0
            if len(same_neighbors) < self.minimum_same_class_neighbors:
                state = "insufficient_class_support"
                recommendation = "REVIEW_CONTEXT"
                reason = "Too few same-class neighbors to judge this assignment reliably."
            elif nearest and score >= self.disagreement_threshold:
                state = "review"
                recommendation = "REVIEW"
                reason = "Most available top-k neighbors do not share the assigned category set."
            else:
                state = "not_flagged"
                recommendation = "NO_AUTOMATIC_ACTION"
                reason = "The configured neighborhood disagreement threshold was not reached."
            findings.append(LabelManipulationFinding(
                image_id=image_id,
                file_path=image.file_path,
                assigned_labels=assigned,
                nearest_neighbors=nearest,
                nearest_neighbor_class_distribution=distribution,
                same_class_neighbor_count=len(same_neighbors),
                different_class_neighbor_count=different_count,
                strongest_competing_label=strongest,
                similarity_to_same_class_neighbors=same_similarity,
                similarity_to_strongest_competing_class=strongest_similarity,
                label_inconsistency_score=score,
                configured_threshold=self.disagreement_threshold,
                embedding_space_id=space_id,
                state=state,
                recommendation=recommendation,
                state_reason=reason,
                source_metadata=_source_metadata(dataset, image),
            ))

        findings.sort(key=lambda finding: _id_key(finding.image_id))
        excluded.sort(key=lambda entry: _id_key(entry.image_id))
        suspicious_count = sum(finding.state == "review" for finding in findings)
        return LabelManipulationResult(
            schema_version=LABEL_MANIPULATION_SCHEMA_VERSION,
            detector_name=LABEL_MANIPULATION_DETECTOR_NAME,
            detector_version=LABEL_MANIPULATION_DETECTOR_VERSION,
            configuration=LabelManipulationConfiguration(
                self.k, self.disagreement_threshold, self.minimum_same_class_neighbors,
            ),
            embedding_space_id=space_id,
            total_images_analyzed=len(findings),
            suspicious_count=suspicious_count,
            findings=tuple(findings),
            excluded_images=tuple(excluded),
            dataset_source_metadata=source_metadata,
        )

    @staticmethod
    def _category_name(dataset: Dataset, category_id: int, names: set[str]) -> Optional[str]:
        mapped = dataset.categories.get(category_id)
        if isinstance(mapped, str) and mapped.strip():
            return mapped.strip()
        return sorted(names)[0] if names else None

    @staticmethod
    def _validate_feature_batch(dataset: Dataset, batch: FeatureBatch) -> str:
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
        if expected_space != space_id:
            raise EmbeddingSpaceMismatchError("FeatureBatch embedding_space_id does not match its metadata")
        if isinstance(batch.embedding_dim, bool) or not isinstance(batch.embedding_dim, int) or batch.embedding_dim <= 0:
            raise InvalidFeatureBatchError("FeatureBatch embedding_dim must be a positive integer")
        feature_ids: set[ImageId] = set()
        dataset_ids = set()
        for image in dataset.images:
            _id_key(image.image_id)
            if image.image_id in dataset_ids:
                raise DatasetFeatureMismatchError(f"Duplicate dataset image_id: {image.image_id!r}")
            dataset_ids.add(image.image_id)
        for embedding in batch.embeddings:
            if not isinstance(embedding, ImageEmbedding):
                raise InvalidFeatureBatchError("FeatureBatch embeddings must contain ImageEmbedding records")
            _id_key(embedding.image_id)
            if embedding.image_id in feature_ids:
                raise DatasetFeatureMismatchError(f"Duplicate embedding image_id: {embedding.image_id!r}")
            feature_ids.add(embedding.image_id)
            record_space = getattr(embedding, "embedding_space_id", space_id)
            if record_space != space_id:
                raise EmbeddingSpaceMismatchError(
                    f"Embedding {embedding.image_id!r} belongs to {record_space!r}, expected {space_id!r}"
                )
            vector = np.asarray(embedding.vector)
            if vector.ndim != 1 or vector.size != batch.embedding_dim:
                raise InvalidFeatureBatchError(
                    f"Embedding {embedding.image_id!r} dimension {vector.size} does not match {batch.embedding_dim}"
                )
            if not (np.issubdtype(vector.dtype, np.integer) or np.issubdtype(vector.dtype, np.floating)):
                raise InvalidFeatureBatchError(f"Embedding {embedding.image_id!r} must be real numeric values")
            if not np.isfinite(vector).all():
                raise InvalidFeatureBatchError(f"Embedding {embedding.image_id!r} contains NaN or infinite values")
            if embedding.image_id not in dataset_ids:
                raise DatasetFeatureMismatchError(
                    f"FeatureBatch contains image_id {embedding.image_id!r} absent from the Dataset"
                )
        return space_id


__all__ = [
    "ContributorSourceMetadata",
    "DatasetFeatureMismatchError",
    "DEFAULT_DISAGREEMENT_THRESHOLD",
    "DEFAULT_MIN_SAME_CLASS_NEIGHBORS",
    "DEFAULT_NEIGHBOR_COUNT",
    "ExcludedImage",
    "LabelManipulationConfiguration",
    "LabelManipulationDetector",
    "LabelManipulationError",
    "LabelManipulationFinding",
    "LabelManipulationResult",
    "LabelReference",
    "LABEL_MANIPULATION_DETECTOR_NAME",
    "LABEL_MANIPULATION_DETECTOR_VERSION",
    "LABEL_MANIPULATION_SCHEMA_VERSION",
    "MissingEmbeddingError",
    "NeighborClassCount",
    "NeighborEvidence",
]
