from pathlib import Path

import numpy as np
import pytest

from trinetra.detectors import (
    EmbeddingSpaceMismatchError,
    InvalidFeatureBatchError,
    OODDetectionResult,
    OODDetector,
)
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure


def _batch(name, entries, *, dim=2, model_id="mock-v1", weights="sha256:mock", failures=()):
    embeddings = [
        ImageEmbedding(
            image_id,
            f"{image_id}.jpg",
            np.asarray(vector, dtype=np.float32),
            Path(name) / f"{image_id}.jpg",
        )
        for image_id, vector in entries
    ]
    return FeatureBatch(
        dataset_name=name,
        backend="mock",
        model_id=model_id,
        weights_identity=weights,
        embedding_dim=dim,
        device="cpu",
        embeddings=embeddings,
        failures=list(failures),
        preprocessing={"method": "synthetic"},
        normalization={"method": "l2"},
        input_fingerprint=f"sha256:{name}-fingerprint",
    )


def test_query_identical_to_reference_is_in_distribution():
    reference = _batch("known-good", [(1, [1, 0])])
    query = _batch("new-batch", [(100, [1, 0])])
    result = OODDetector().detect(reference, query)

    assert isinstance(result, OODDetectionResult)
    finding = result.findings[0]
    assert finding.nearest_reference_similarity == pytest.approx(1.0)
    assert finding.state == "in_distribution"
    assert finding.anomaly_score == 0.0
    assert finding.recommendation == "NO_AUTOMATIC_ACTION"


def test_distant_query_is_distribution_shift_candidate_not_attack_claim():
    reference = _batch("reference", [(1, [1, 0]), (2, [0.9, 0.1])])
    query = _batch("query", [(50, [-1, 0])])
    result = OODDetector().detect(reference, query)
    finding = result.findings[0]

    assert finding.state == "distribution_shift_candidate"
    assert finding.recommendation == "REVIEW"
    assert finding.interpretation == "Feature representation is distant from the reference distribution."
    assert finding.nearest_reference_similarity < result.threshold
    assert finding.anomaly_score == pytest.approx(result.threshold - finding.nearest_reference_similarity)
    assert "malicious" not in finding.interpretation.lower()
    assert "poisoned" not in finding.interpretation.lower()


def test_threshold_is_configurable_and_inclusive_on_in_distribution_side():
    reference = _batch("reference", [(1, [1, 0])])
    query = _batch("query", [(2, [0.8, 0.6])])
    low = OODDetector(threshold=0.7).detect(reference, query)
    high = OODDetector(threshold=0.9).detect(reference, query)
    assert low.findings[0].state == "in_distribution"
    assert high.findings[0].state == "distribution_shift_candidate"
    with pytest.raises(ValueError, match="threshold"):
        OODDetector(1.1)


def test_nearest_reference_is_selected_from_multiple_candidates():
    reference = _batch("reference", [(9, [1, 0]), (3, [0, 1]), (10, [0.1, 0.99])])
    query = _batch("query", [(5, [0, 1])])
    finding = OODDetector().detect(reference, query).findings[0]
    assert finding.nearest_reference_image_ids == (3,)
    assert finding.nearest_reference_similarity == pytest.approx(1.0)
    assert finding.nearest_reference_neighbors[0].file_path == Path("reference/3.jpg")


def test_equal_similarity_ties_use_stable_reference_id_order():
    reference = _batch("reference", [(9, [1, 0]), (2, [1, 0])])
    query = _batch("query", [(7, [1, 0])])
    finding = OODDetector().detect(reference, query).findings[0]
    assert finding.nearest_reference_image_ids == (2,)


def test_multiple_reference_images_and_query_findings_are_deterministic():
    reference = _batch("reference", [(1, [1, 0]), (2, [0, 1])])
    query = _batch("query", [(8, [0, 1]), (4, [1, 0])])
    detector = OODDetector()
    first = detector.detect(reference, query)
    second = detector.detect(reference, query)
    assert first == second
    assert [finding.image_id for finding in first.findings] == [4, 8]


def test_empty_reference_set_returns_unscored_query_not_false_anomaly():
    reference = _batch("empty-reference", [])
    query = _batch("query", [(4, [1, 0])])
    result = OODDetector().detect(reference, query)
    assert result.findings == ()
    assert result.anomalous_count == 0
    assert result.unscored_queries[0].reason_code == "empty_reference_set"


def test_empty_query_batch_returns_empty_result():
    reference = _batch("reference", [(1, [1, 0])])
    query = _batch("empty-query", [])
    result = OODDetector().detect(reference, query)
    assert result.total_query_images == 0
    assert result.findings == ()
    assert result.unscored_queries == ()


def test_zero_vector_has_safe_zero_cosine_and_thresholded_result():
    reference = _batch("reference", [(1, [0, 0])])
    query = _batch("query", [(2, [0, 0])])
    finding = OODDetector().detect(reference, query).findings[0]
    assert finding.nearest_reference_similarity == 0.0
    assert finding.state == "distribution_shift_candidate"
    assert np.isfinite(finding.anomaly_score)


@pytest.mark.parametrize("bad_vector", [np.array([np.nan, 0]), np.array([np.inf, 1])])
def test_invalid_nonfinite_vectors_are_rejected(bad_vector):
    reference = _batch("reference", [(1, [1, 0])])
    query = _batch("query", [(2, [0, 1])])
    query.embeddings[0].vector = bad_vector
    with pytest.raises(InvalidFeatureBatchError, match="NaN or infinite"):
        OODDetector().detect(reference, query)


def test_reference_and_query_dimension_mismatch_is_rejected():
    reference = _batch("reference", [(1, [1, 0])], dim=2)
    query = _batch("query", [(2, [1, 0, 0])], dim=3)
    with pytest.raises(InvalidFeatureBatchError, match="dimensions differ"):
        OODDetector().detect(reference, query)


def test_mixed_embedding_spaces_are_rejected():
    reference = _batch("reference", [(1, [1, 0])], model_id="dinov2")
    query = _batch("query", [(2, [1, 0])], model_id="resnet50")
    with pytest.raises(EmbeddingSpaceMismatchError, match="different embedding spaces"):
        OODDetector().detect(reference, query)


def test_reference_batch_mutated_to_an_inconsistent_space_is_rejected():
    reference = _batch("reference", [(1, [1, 0])])
    query = _batch("query", [(2, [1, 0])])
    reference.embedding_space_id = "sha256:wrong-space"
    with pytest.raises(EmbeddingSpaceMismatchError, match="does not match"):
        OODDetector().detect(reference, query)


def test_query_id_present_in_reference_is_excluded_to_prevent_self_similarity():
    reference = _batch("reference", [(1, [1, 0]), (2, [-1, 0])])
    query = _batch("query", [(1, [1, 0])])
    finding = OODDetector().detect(reference, query).findings[0]
    assert finding.nearest_reference_image_ids == (2,)
    assert finding.nearest_reference_similarity == pytest.approx(-1.0)
    assert finding.state == "distribution_shift_candidate"


def test_query_overlapping_only_reference_is_unscored():
    reference = _batch("reference", [(1, [1, 0])])
    query = _batch("query", [(1, [1, 0])])
    result = OODDetector().detect(reference, query)
    assert result.findings == ()
    assert result.unscored_queries[0].reason_code == "no_comparable_reference"


def test_layer1b_query_failures_are_reported_as_unscored():
    reference = _batch("reference", [(1, [1, 0])])
    failure = ImageFailure(2, Path("query/2.jpg"), "image_unreadable", "corrupt image")
    query = _batch("query", [], failures=[failure])
    result = OODDetector().detect(reference, query)
    assert result.total_query_images == 1
    assert result.findings == ()
    assert result.unscored_queries[0].reason_code == "image_unreadable"
    assert "corrupt image" in result.unscored_queries[0].reason


def test_result_schema_evidence_and_reference_set_provenance():
    reference = _batch("known-good-v3", [(1, [1, 0])])
    query = _batch("incoming", [(90, [0, 1])])
    result = OODDetector(threshold=0.75).detect(
        reference, query, reference_set_id="approved-reference-2026-10"
    )
    finding = result.findings[0]
    assert result.schema_version == 1
    assert result.detector_name == "reference_cosine_distribution_shift"
    assert result.detector_version == "1.0.0"
    assert result.threshold == 0.75
    assert result.reference_set.identifier == "approved-reference-2026-10"
    assert result.reference_set.identifier_source == "caller"
    assert result.reference_set.dataset_name == "known-good-v3"
    assert result.reference_set.input_fingerprint == "sha256:known-good-v3-fingerprint"
    assert result.embedding_space_id == reference.embedding_space_id
    assert result.total_query_images == 1
    assert result.anomalous_count == 1
    assert finding.image_id == 90
    assert finding.file_path == Path("incoming/90.jpg")
    assert finding.nearest_reference_image_ids == (1,)
    assert finding.nearest_reference_similarity == pytest.approx(0.0)
    assert finding.configured_threshold == 0.75
    assert finding.embedding_space_id == reference.embedding_space_id
    assert finding.reference_set_identifier == result.reference_set.identifier


def test_reference_identifier_defaults_to_available_dataset_name():
    reference = _batch("named-reference", [(1, [1, 0])])
    query = _batch("query", [(2, [1, 0])])
    result = OODDetector().detect(reference, query)
    assert result.reference_set.identifier == "named-reference"
    assert result.reference_set.identifier_source == "feature_batch.dataset_name"
