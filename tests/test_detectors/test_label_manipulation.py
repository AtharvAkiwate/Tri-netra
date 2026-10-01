from pathlib import Path

import numpy as np
import pytest

from trinetra.detectors import (
    DatasetFeatureMismatchError,
    EmbeddingSpaceMismatchError,
    InvalidFeatureBatchError,
    LabelManipulationDetector,
    LabelManipulationResult,
    MissingEmbeddingError,
)
from trinetra.features.models import FeatureBatch, ImageEmbedding
from trinetra.parsers.models import Annotation, BoundingBox, Dataset, DatasetFormat, DatasetImage


CATEGORIES = {11: "sparrow", 29: "aircraft", 47: "vehicle"}


def _inputs(specs, *, embedding_ids=None):
    """specs are (image_id, category_id-or-None, vector-or-None)."""
    images = []
    embeddings = []
    for image_id, category_id, vector in specs:
        annotations = []
        if category_id is not None:
            annotations = [Annotation(
                annotation_id=f"ann-{image_id}",
                category_id=category_id,
                bbox=BoundingBox(0, 0, 1, 1),
                category_name=CATEGORIES.get(category_id),
            )]
        images.append(DatasetImage(
            image_id=image_id,
            file_name=f"{image_id}.jpg",
            width=32,
            height=32,
            annotations=annotations,
            file_path=Path("dataset") / f"{image_id}.jpg",
        ))
        if vector is not None and (embedding_ids is None or image_id in embedding_ids):
            embeddings.append(ImageEmbedding(
                image_id=image_id,
                file_name=f"{image_id}.jpg",
                vector=np.asarray(vector, dtype=np.float32),
                file_path=Path("dataset") / f"{image_id}.jpg",
            ))
    dataset = Dataset("synthetic-labels", DatasetFormat.COCO, images, dict(CATEGORIES))
    batch = FeatureBatch(
        dataset_name=dataset.name,
        backend="mock",
        model_id="synthetic-v1",
        weights_identity="sha256:synthetic",
        embedding_dim=2,
        device="cpu",
        embeddings=embeddings,
        preprocessing={"method": "synthetic"},
        normalization={"method": "l2"},
    )
    return dataset, batch


def _two_class_specs():
    return [
        (1, 11, [1.0, 0.0]),
        (2, 11, [0.99, 0.1]),
        (3, 11, [0.98, -0.1]),
        (4, 29, [0.0, 1.0]),
        (5, 29, [0.1, 0.99]),
        (6, 29, [-0.1, 0.98]),
    ]


def _cross_class_specs():
    return [
        (1, 11, [1.0, 0.0]),
        (2, 11, [0.985, -0.174]),
        # Assigned sparrow, but embedded with the aircraft neighborhood.
        (3, 11, [0.94, 0.342]),
        (4, 29, [0.946, 0.326]),
        (5, 29, [0.94, 0.342]),
        (6, 29, [0.0, 1.0]),
    ]


def test_obvious_same_class_neighborhood_is_not_suspicious():
    dataset, batch = _inputs(_two_class_specs())
    result = LabelManipulationDetector(k=3).detect(dataset, batch)

    sparrow = next(f for f in result.findings if f.image_id == 1)
    assert sparrow.state == "not_flagged"
    assert sparrow.same_class_neighbor_count == 2
    assert sparrow.different_class_neighbor_count == 1
    assert sparrow.label_inconsistency_score == pytest.approx(1 / 3)
    assert result.suspicious_count == 0


def test_cross_class_neighborhood_is_flagged_with_competing_class_evidence():
    dataset, batch = _inputs(_cross_class_specs())
    result = LabelManipulationDetector(k=3, disagreement_threshold=0.6).detect(dataset, batch)
    finding = next(f for f in result.findings if f.image_id == 3)

    assert finding.state == "review"
    assert finding.recommendation == "REVIEW"
    assert finding.assigned_labels[0].category_name == "sparrow"
    assert finding.strongest_competing_label.category_id == 29
    assert finding.strongest_competing_label.category_name == "aircraft"
    assert finding.label_inconsistency_score == pytest.approx(2 / 3)
    assert finding.similarity_to_same_class_neighbors is not None
    assert finding.similarity_to_strongest_competing_class > finding.similarity_to_same_class_neighbors


def test_configurable_k_and_fewer_than_k_available_neighbors():
    dataset, batch = _inputs(_two_class_specs()[:2])
    result = LabelManipulationDetector(k=5).detect(dataset, batch)
    assert len(result.findings[0].nearest_neighbors) == 1
    assert result.configuration.k == 5


def test_neighbor_ordering_and_results_are_deterministic_with_stable_ties():
    specs = [(20, 11, [1, 0]), (3, 29, [1, 0]), (10, 47, [1, 0])]
    dataset, batch = _inputs(specs)
    first = LabelManipulationDetector(k=2, minimum_same_class_neighbors=0).detect(dataset, batch)
    second = LabelManipulationDetector(k=2, minimum_same_class_neighbors=0).detect(dataset, batch)
    assert first == second
    finding = next(f for f in first.findings if f.image_id == 20)
    assert [neighbor.image_id for neighbor in finding.nearest_neighbors] == [10, 3]


def test_threshold_configuration_changes_review_state():
    dataset, batch = _inputs(_cross_class_specs())
    low = LabelManipulationDetector(k=3, disagreement_threshold=0.6).detect(dataset, batch)
    high = LabelManipulationDetector(k=3, disagreement_threshold=0.9).detect(dataset, batch)
    assert next(f for f in low.findings if f.image_id == 3).state == "review"
    assert next(f for f in high.findings if f.image_id == 3).state == "not_flagged"
    assert high.configuration.disagreement_threshold == 0.9


def test_missing_required_labeled_embedding_raises_explicit_error():
    dataset, batch = _inputs(_two_class_specs(), embedding_ids={1, 2, 3, 4, 5})
    with pytest.raises(MissingEmbeddingError, match="Labeled image 6"):
        LabelManipulationDetector().detect(dataset, batch)


def test_mixed_embedding_spaces_are_rejected():
    dataset, batch = _inputs(_two_class_specs()[:2])
    batch.embeddings[1].embedding_space_id = "sha256:other-space"
    with pytest.raises(EmbeddingSpaceMismatchError, match="belongs to"):
        LabelManipulationDetector().detect(dataset, batch)


def test_embedding_space_metadata_mismatch_is_rejected():
    dataset, batch = _inputs(_two_class_specs()[:2])
    batch.embedding_space_id = "sha256:other-space"
    with pytest.raises(EmbeddingSpaceMismatchError, match="does not match"):
        LabelManipulationDetector().detect(dataset, batch)


def test_dimension_mismatch_is_rejected():
    dataset, batch = _inputs(_two_class_specs()[:2])
    batch.embeddings[0].vector = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    with pytest.raises(InvalidFeatureBatchError, match="dimension"):
        LabelManipulationDetector().detect(dataset, batch)


@pytest.mark.parametrize("invalid", [np.array([np.nan, 0]), np.array([np.inf, 1])])
def test_nonfinite_vectors_are_rejected(invalid):
    dataset, batch = _inputs(_two_class_specs()[:2])
    batch.embeddings[0].vector = invalid
    with pytest.raises(InvalidFeatureBatchError, match="NaN or infinite"):
        LabelManipulationDetector().detect(dataset, batch)


def test_empty_dataset_returns_versioned_empty_result():
    dataset, batch = _inputs([])
    result = LabelManipulationDetector().detect(dataset, batch)
    assert isinstance(result, LabelManipulationResult)
    assert result.schema_version == 1
    assert result.detector_name == "label_manipulation_neighborhood"
    assert result.detector_version == "1.0.0"
    assert result.total_images_analyzed == 0
    assert result.suspicious_count == 0
    assert result.findings == ()


def test_single_image_has_no_neighbors_and_insufficient_support():
    dataset, batch = _inputs([(1, 11, [1, 0])])
    result = LabelManipulationDetector().detect(dataset, batch)
    finding = result.findings[0]
    assert finding.nearest_neighbors == ()
    assert finding.state == "insufficient_class_support"
    assert result.suspicious_count == 0


def test_single_member_class_is_not_automatically_flagged():
    specs = [(1, 11, [1, 0]), (2, 29, [0.99, 0.01]), (3, 29, [0.98, 0.02])]
    dataset, batch = _inputs(specs)
    result = LabelManipulationDetector(k=2, disagreement_threshold=0.6).detect(dataset, batch)
    finding = next(f for f in result.findings if f.image_id == 1)
    assert finding.state == "insufficient_class_support"
    assert result.suspicious_count == 0


def test_unlabeled_and_unresolvable_category_images_are_reported_as_excluded():
    specs = [(1, None, [1, 0])]
    dataset, batch = _inputs(specs)
    dataset.images[0].annotations = [Annotation("missing-cat", None, BoundingBox(0, 0, 1, 1))]
    result = LabelManipulationDetector().detect(dataset, batch)
    assert result.total_images_analyzed == 0
    assert result.excluded_images[0].exclusion_code == "missing_category"


def test_evidence_contains_ids_paths_labels_similarity_score_threshold_and_space():
    dataset, batch = _inputs(_cross_class_specs())
    result = LabelManipulationDetector(k=3, disagreement_threshold=0.6).detect(dataset, batch)
    finding = next(f for f in result.findings if f.image_id == 3)
    assert finding.file_path == Path("dataset/3.jpg")
    assert finding.assigned_labels[0].category_id == 11
    assert finding.nearest_neighbors[0].image_id in {4, 5}
    assert finding.nearest_neighbors[0].labels[0].category_name == "aircraft"
    assert -1.0 <= finding.nearest_neighbors[0].similarity <= 1.0
    assert finding.configured_threshold == 0.6
    assert finding.embedding_space_id == batch.embedding_space_id
    assert finding.same_class_neighbor_count + finding.different_class_neighbor_count == 3


def test_available_contributor_source_and_batch_metadata_is_preserved():
    dataset, batch = _inputs(_two_class_specs()[:2])
    dataset.contributor_id = "team-17"
    dataset.images[0].source_id = "camera-source-A"
    dataset.images[0].batch_id = 42
    result = LabelManipulationDetector(k=1).detect(dataset, batch)
    assert result.dataset_source_metadata.contributor == "team-17"
    first = next(f for f in result.findings if f.image_id == 1)
    assert first.source_metadata.contributor == "team-17"
    assert first.source_metadata.source == "camera-source-A"
    assert first.source_metadata.batch == 42
    assert next(f for f in result.findings if f.image_id == 2).source_metadata.source is None


def test_unavailable_contributor_identity_is_not_fabricated():
    dataset, batch = _inputs(_two_class_specs()[:2])
    result = LabelManipulationDetector(k=1).detect(dataset, batch)
    assert result.dataset_source_metadata.contributor is None
    assert result.dataset_source_metadata.source is None
    assert result.dataset_source_metadata.batch is None


def test_feature_ids_not_in_dataset_are_rejected():
    dataset, batch = _inputs(_two_class_specs()[:2])
    batch.embeddings[1].image_id = 999
    with pytest.raises(DatasetFeatureMismatchError, match="absent from the Dataset"):
        LabelManipulationDetector().detect(dataset, batch)


def test_zero_vectors_are_handled_without_invalid_scores():
    dataset, batch = _inputs([(1, 11, [0, 0]), (2, 11, [1, 0])])
    result = LabelManipulationDetector(k=1).detect(dataset, batch)
    assert all(np.isfinite(f.label_inconsistency_score) for f in result.findings)
    assert result.findings[0].nearest_neighbors[0].similarity == 0.0
