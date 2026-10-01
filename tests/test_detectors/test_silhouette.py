import numpy as np
import pytest

from trinetra.detectors.silhouette import (
    SilhouetteDetector,
    SilhouetteEmbeddingSpaceError,
    SilhouetteValidationError,
)
from trinetra.features.models import FeatureBatch, ImageEmbedding
from trinetra.parsers.models import (
    Annotation,
    BoundingBox,
    Dataset,
    DatasetFormat,
    DatasetImage,
)


def _dataset(labels, *, categories=None):
    images = []
    for index, label_set in enumerate(labels):
        annotations = [
            Annotation(index * 10 + offset, category_id, BoundingBox(0, 0, 1, 1),
                       category_name=(categories or {}).get(category_id))
            for offset, category_id in enumerate(label_set)
        ]
        images.append(DatasetImage(index, f"{index}.jpg", 10, 10, annotations))
    return Dataset("synthetic", DatasetFormat.COCO, images, categories or {})


def _batch(dataset, vectors):
    return FeatureBatch(
        dataset_name=dataset.name,
        backend="synthetic",
        model_id="synthetic-v1",
        weights_identity="sha256:synthetic",
        embedding_dim=len(vectors[0]) if vectors else 2,
        device="cpu",
        embeddings=[ImageEmbedding(image.image_id, image.file_name, np.asarray(vector, dtype=np.float32))
                    for image, vector in zip(dataset.images, vectors)],
        preprocessing={"method": "synthetic"},
        normalization={"method": "l2"},
    )


def _separated():
    dataset = _dataset([(1,), (1,), (2,), (2,)], categories={1: "left", 2: "right"})
    batch = _batch(dataset, [[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]])
    return dataset, batch


def test_separated_classes_have_positive_silhouettes_and_summary():
    dataset, batch = _separated()
    result = SilhouetteDetector().analyze(dataset, batch)
    assert result.scorable_images == 4
    assert all(item.silhouette_score > 0.9 for item in result.findings)
    assert result.class_count == 2
    assert [item.sample_count for item in result.per_class_summary] == [2, 2]
    assert result.dataset_mean_silhouette > 0.9
    assert result.dataset_median_silhouette > 0.9
    assert result.min_silhouette <= result.max_silhouette


def test_overlapping_classes_have_lower_score_and_competing_class():
    dataset = _dataset([(1,), (1,), (2,), (2,)])
    batch = _batch(dataset, [[1, 0], [0.99, 0.1], [1, 0], [0.99, 0.1]])
    result = SilhouetteDetector().analyze(dataset, batch)
    assert all(item.silhouette_score < 0.1 for item in result.findings)
    assert result.findings[0].nearest_competing_class.category_ids == (2,)


def test_single_class_and_empty_dataset_are_unscorable_not_errors():
    dataset = _dataset([(1,), (1,)])
    result = SilhouetteDetector().analyze(dataset, _batch(dataset, [[1, 0], [0, 1]]))
    assert result.scorable_images == 0
    assert {item.reason_code for item in result.unscorable_records} == {"no_competing_class"}
    empty = _dataset([])
    empty_result = SilhouetteDetector().analyze(empty, _batch(empty, []))
    assert empty_result.total_images == 0
    assert empty_result.dataset_mean_silhouette is None


def test_single_member_class_is_excluded_with_reason():
    dataset = _dataset([(1,), (1,), (2,)])
    result = SilhouetteDetector().analyze(dataset, _batch(dataset, [[1, 0], [0.9, 0.1], [0, 1]]))
    assert result.scorable_images == 2
    excluded = [item for item in result.unscorable_records if item.reason_code == "class_too_small"]
    assert [item.image_id for item in excluded] == [2]


def test_missing_labels_and_embeddings_are_separately_reported():
    dataset = _dataset([(1,), (1,), (2,), ()])
    batch = _batch(dataset, [[1, 0], [0.9, 0.1]])
    result = SilhouetteDetector().analyze(dataset, batch)
    codes = {item.image_id: item.reason_code for item in result.unscorable_records}
    assert codes[2] == "missing_embedding"
    assert codes[3] == "missing_label"


def test_zero_invalid_and_dimension_mismatch_vectors_are_unscorable():
    dataset, batch = _separated()
    batch.embeddings[0].vector = np.zeros(2, dtype=np.float32)
    batch.embeddings[1].vector = np.array([np.nan, 1], dtype=np.float32)
    batch.embeddings[2].vector = np.ones(3, dtype=np.float32)
    result = SilhouetteDetector().analyze(dataset, batch)
    codes = {item.image_id: item.reason_code for item in result.unscorable_records}
    assert codes[0] == "zero_vector"
    assert codes[1] == "invalid_vector"
    assert codes[2] == "dimension_mismatch"


def test_duplicate_dataset_ids_rejected():
    dataset, batch = _separated()
    dataset.images[1].image_id = dataset.images[0].image_id
    with pytest.raises(SilhouetteValidationError, match="Duplicate"):
        SilhouetteDetector().analyze(dataset, batch)


def test_inconsistent_embedding_space_identity_rejected():
    dataset, batch = _separated()
    batch.embedding_space_id = "sha256:other-space"
    with pytest.raises(SilhouetteEmbeddingSpaceError, match="does not match"):
        SilhouetteDetector().analyze(dataset, batch)


def test_multilabel_policy_uses_sorted_complete_category_set():
    dataset = _dataset([(2, 1), (1, 2), (1,), (1,), (2,), (2,)])
    batch = _batch(dataset, [[1, 0], [0.9, 0.1], [1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]])
    result = SilhouetteDetector().analyze(dataset, batch)
    assert result.findings[0].assigned_class.category_ids == (1, 2)
    assert result.findings[1].assigned_class.category_ids == (1, 2)
    assert all(item.assigned_class.category_ids != (1,) for item in result.findings[:2])


def test_deterministic_ordering_schema_and_evidence_fields():
    dataset, batch = _separated()
    first = SilhouetteDetector().analyze(dataset, batch)
    second = SilhouetteDetector().analyze(dataset, batch)
    assert first == second
    assert [item.image_id for item in first.findings] == [0, 1, 2, 3]
    assert first.schema_version == 1
    assert first.detector_name == "label_silhouette_validation"
    assert first.embedding_space_id == batch.embedding_space_id
    assert first.distance_metric == "cosine"
    assert first.findings[0].file_path is None
    assert "Feature representation" in first.findings[0].interpretation
    assert "not, by itself" in first.limitations[0]


def test_minimum_class_size_and_metric_configuration():
    with pytest.raises(SilhouetteValidationError):
        SilhouetteDetector(minimum_class_size=1)
    with pytest.raises(SilhouetteValidationError):
        SilhouetteDetector(distance_metric="euclidean")
    dataset, batch = _separated()
    result = SilhouetteDetector(minimum_class_size=3).analyze(dataset, batch)
    assert result.minimum_class_size == 3
    assert result.scorable_images == 0
