"""Silhouette validation of Layer 1B features against Layer 1A labels."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from trinetra.detectors.near_duplicate import cosine_similarity
from trinetra.features.models import FeatureBatch, ImageId, make_embedding_space_id
from trinetra.parsers.models import Dataset, DatasetImage

SILHOUETTE_SCHEMA_VERSION = 1
SILHOUETTE_DETECTOR_NAME = "label_silhouette_validation"
SILHOUETTE_DETECTOR_VERSION = "1.0.0"
DEFAULT_MINIMUM_CLASS_SIZE = 2


class SilhouetteValidationError(ValueError):
    """Invalid dataset, FeatureBatch, or detector configuration."""


class SilhouetteEmbeddingSpaceError(SilhouetteValidationError):
    """The supplied batch does not describe one valid embedding space."""


@dataclass(frozen=True)
class SilhouetteConfiguration:
    minimum_class_size: int
    distance_metric: str = "cosine"


@dataclass(frozen=True)
class SilhouetteClass:
    """Class represented by a sorted, potentially multi-label category set."""

    category_ids: tuple[int, ...]
    category_names: tuple[Optional[str], ...]


@dataclass(frozen=True)
class SilhouetteFinding:
    image_id: ImageId
    file_path: Optional[Path]
    assigned_class: SilhouetteClass
    a_mean_intra_class_distance: float
    b_mean_nearest_competing_class_distance: float
    silhouette_score: float
    nearest_competing_class: SilhouetteClass
    interpretation: str


@dataclass(frozen=True)
class UnscorableImage:
    image_id: ImageId
    file_path: Optional[Path]
    reason_code: str
    reason: str


@dataclass(frozen=True)
class PerClassSilhouetteSummary:
    assigned_class: SilhouetteClass
    sample_count: int
    mean_silhouette_score: float


@dataclass(frozen=True)
class SilhouetteResult:
    schema_version: int
    detector_name: str
    detector_version: str
    embedding_space_id: str
    distance_metric: str
    minimum_class_size: int
    total_images: int
    scorable_images: int
    unscorable_images: int
    class_count: int
    dataset_mean_silhouette: Optional[float]
    dataset_median_silhouette: Optional[float]
    min_silhouette: Optional[float]
    max_silhouette: Optional[float]
    per_class_summary: tuple[PerClassSilhouetteSummary, ...]
    findings: tuple[SilhouetteFinding, ...]
    unscorable_records: tuple[UnscorableImage, ...]
    limitations: tuple[str, ...]


class SilhouetteDetector:
    """Compute exact cosine-distance silhouettes for labeled embeddings.

    An image's class is the complete sorted set of valid category IDs on that
    image. ``minimum_class_size`` applies to the assigned class so that a(i)
    has peers; a singleton may still serve as a competing class for b(i).
    Results are separation signals for
    analyst review, not calibrated probabilities or poisoning determinations.
    """

    def __init__(self, minimum_class_size: int = DEFAULT_MINIMUM_CLASS_SIZE, distance_metric: str = "cosine") -> None:
        if isinstance(minimum_class_size, bool) or not isinstance(minimum_class_size, int) or minimum_class_size < 2:
            raise SilhouetteValidationError("minimum_class_size must be an integer >= 2")
        if distance_metric != "cosine":
            raise SilhouetteValidationError("Only cosine distance is currently supported")
        self.minimum_class_size = minimum_class_size
        self.distance_metric = distance_metric

    def analyze(self, dataset: Dataset, feature_batch: FeatureBatch) -> SilhouetteResult:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Layer 1A Dataset")
        if not isinstance(feature_batch, FeatureBatch):
            raise TypeError("feature_batch must be a Layer 1B FeatureBatch")
        space_id = feature_batch.embedding_space_id
        if not isinstance(space_id, str) or not space_id.strip():
            raise SilhouetteEmbeddingSpaceError("FeatureBatch must have an embedding_space_id")
        expected_space_id = make_embedding_space_id(
            model_id=feature_batch.model_id,
            weights_identity=feature_batch.weights_identity,
            preprocessing=feature_batch.preprocessing,
            embedding_dim=feature_batch.embedding_dim,
            normalization=feature_batch.normalization,
        )
        if space_id != expected_space_id:
            raise SilhouetteEmbeddingSpaceError(
                "FeatureBatch embedding_space_id does not match its model, weights, preprocessing, dimension, and normalization metadata"
            )
        images: dict[ImageId, DatasetImage] = {}
        for image in dataset.images:
            if isinstance(image.image_id, bool) or not isinstance(image.image_id, (int, str)):
                raise SilhouetteValidationError(f"Unsupported image_id: {image.image_id!r}")
            if image.image_id in images:
                raise SilhouetteValidationError(f"Duplicate image_id in dataset: {image.image_id!r}")
            images[image.image_id] = image
        embeddings = {item.image_id: item for item in feature_batch.embeddings}

        classes: dict[ImageId, SilhouetteClass] = {}
        vectors: dict[ImageId, np.ndarray] = {}
        excluded: list[UnscorableImage] = []
        for image in sorted(dataset.images, key=lambda item: _id_key(item.image_id)):
            labels: dict[int, Optional[str]] = {}
            for annotation in image.annotations:
                category_id = annotation.category_id
                if isinstance(category_id, bool) or not isinstance(category_id, int):
                    continue
                category_name = annotation.category_name or dataset.categories.get(category_id)
                if category_id not in labels or labels[category_id] is None:
                    labels[category_id] = category_name
            if not labels:
                excluded.append(UnscorableImage(image.image_id, image.file_path, "missing_label", "Image has no valid category annotation"))
                continue
            item = embeddings.get(image.image_id)
            if item is None:
                excluded.append(UnscorableImage(image.image_id, image.file_path, "missing_embedding", "No Layer 1B embedding is available"))
                continue
            vector = np.asarray(item.vector)
            if vector.ndim != 1 or vector.size != feature_batch.embedding_dim:
                excluded.append(UnscorableImage(image.image_id, image.file_path, "dimension_mismatch", "Embedding dimension does not match FeatureBatch"))
                continue
            if not (
                np.issubdtype(vector.dtype, np.integer)
                or np.issubdtype(vector.dtype, np.floating)
            ) or not np.isfinite(vector).all():
                excluded.append(UnscorableImage(image.image_id, image.file_path, "invalid_vector", "Embedding contains non-numeric or non-finite values"))
                continue
            if not np.any(vector != 0):
                excluded.append(UnscorableImage(image.image_id, image.file_path, "zero_vector", "Cosine distance is undefined for a zero vector"))
                continue
            category_ids = tuple(sorted(labels))
            names = tuple(labels[category_id] for category_id in category_ids)
            classes[image.image_id] = SilhouetteClass(category_ids, names)
            vectors[image.image_id] = vector

        members_by_class: dict[tuple[int, ...], list[ImageId]] = defaultdict(list)
        class_records: dict[tuple[int, ...], SilhouetteClass] = {}
        for image_id, assigned_class in classes.items():
            key = assigned_class.category_ids
            members_by_class[key].append(image_id)
            class_records[key] = assigned_class
        eligible_keys = {key for key, members in members_by_class.items() if len(members) >= self.minimum_class_size}
        findings: list[SilhouetteFinding] = []
        for image_id in sorted(classes, key=_id_key):
            own_key = classes[image_id].category_ids
            if own_key not in eligible_keys:
                excluded.append(UnscorableImage(image_id, images[image_id].file_path, "class_too_small", f"Assigned class has fewer than {self.minimum_class_size} scorable images"))
                continue
            competing_keys = sorted((key for key in members_by_class if key != own_key))
            if not competing_keys:
                excluded.append(UnscorableImage(image_id, images[image_id].file_path, "no_competing_class", "No other labeled class has a scorable embedding"))
                continue
            own_distances = [self._distance(vectors[image_id], vectors[other_id]) for other_id in members_by_class[own_key] if other_id != image_id]
            if not own_distances:
                excluded.append(UnscorableImage(image_id, images[image_id].file_path, "no_intra_class_peer", "No other scorable image belongs to the assigned class"))
                continue
            a = float(math.fsum(own_distances) / len(own_distances))
            class_distances = {
                key: float(math.fsum(self._distance(vectors[image_id], vectors[other_id]) for other_id in members_by_class[key]) / len(members_by_class[key]))
                for key in competing_keys
            }
            nearest_key = min(competing_keys, key=lambda key: (class_distances[key], key))
            b = class_distances[nearest_key]
            denominator = max(a, b)
            score = 0.0 if denominator == 0.0 else float((b - a) / denominator)
            interpretation = (
                "Feature representation is well separated from competing classes."
                if score > 0.0 else
                "Feature representation overlaps with another class and should be reviewed."
            )
            findings.append(SilhouetteFinding(
                image_id=image_id,
                file_path=images[image_id].file_path,
                assigned_class=classes[image_id],
                a_mean_intra_class_distance=a,
                b_mean_nearest_competing_class_distance=b,
                silhouette_score=score,
                nearest_competing_class=class_records[nearest_key],
                interpretation=interpretation,
            ))
        findings.sort(key=lambda item: _id_key(item.image_id))
        excluded.sort(key=lambda item: _id_key(item.image_id))
        by_class: dict[tuple[int, ...], list[float]] = defaultdict(list)
        for finding in findings:
            by_class[finding.assigned_class.category_ids].append(finding.silhouette_score)
        summaries = tuple(
            PerClassSilhouetteSummary(
                class_records[key], len(scores), float(math.fsum(scores) / len(scores))
            )
            for key, scores in sorted(by_class.items())
        )
        scores = [item.silhouette_score for item in findings]
        return SilhouetteResult(
            schema_version=SILHOUETTE_SCHEMA_VERSION,
            detector_name=SILHOUETTE_DETECTOR_NAME,
            detector_version=SILHOUETTE_DETECTOR_VERSION,
            embedding_space_id=space_id,
            distance_metric=self.distance_metric,
            minimum_class_size=self.minimum_class_size,
            total_images=len(dataset.images),
            scorable_images=len(findings),
            unscorable_images=len(excluded),
            class_count=len(summaries),
            dataset_mean_silhouette=float(math.fsum(scores) / len(scores)) if scores else None,
            dataset_median_silhouette=float(np.median(np.asarray(scores, dtype=np.float64))) if scores else None,
            min_silhouette=min(scores) if scores else None,
            max_silhouette=max(scores) if scores else None,
            per_class_summary=summaries,
            findings=tuple(findings),
            unscorable_records=tuple(excluded),
            limitations=(
                "Silhouette analysis measures feature-space separation and is not, by itself, evidence of malicious manipulation or poisoning.",
                "Low or negative scores may reflect legitimate class overlap, difficult samples, poor embeddings, insufficient reference data, distribution shift, or incorrect class assumptions.",
                "Exact comparisons use O(n^2) time and retain per-image distances only; no full pairwise matrix is built.",
            ),
        )

    @staticmethod
    def _distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        try:
            return 1.0 - cosine_similarity(vector_a, vector_b)
        except (ValueError, FloatingPointError) as exc:
            raise SilhouetteValidationError(f"Could not calculate cosine distance: {exc}") from exc


def _id_key(image_id: ImageId) -> tuple[str, str]:
    return ("int" if isinstance(image_id, int) else "str", str(image_id))
