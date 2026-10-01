from pathlib import Path
from types import SimpleNamespace

import pytest

from trinetra.contributor import (
    AggregationWeights,
    ContributorRiskAggregator,
    InvalidDetectorEvidenceError,
    InvalidProvenanceError,
    ProvenanceMetadata,
    Recommendation,
    UnknownDetectorError,
)
from trinetra.detectors.label_manipulation import (
    LABEL_MANIPULATION_DETECTOR_NAME,
    ContributorSourceMetadata,
    LabelManipulationConfiguration,
    LabelManipulationFinding,
    LabelManipulationResult,
    LabelReference,
)
from trinetra.detectors.near_duplicate import DuplicatePair, NearDuplicateResult
from trinetra.detectors.ood import (
    OODDetectionResult,
    OODFinding,
    ReferenceSetMetadata,
)
from trinetra.parsers.models import Dataset, DatasetFormat, DatasetImage


def _dataset(ids=(1, 2, 3, 4)):
    images = [DatasetImage(image_id, f"{image_id}.jpg", 20, 20, file_path=Path(f"data/{image_id}.jpg")) for image_id in ids]
    return Dataset("risk-data", DatasetFormat.COCO, images)


def _near_result(pairs=((1, 2, 0.98), (2, 3, 0.96))):
    pair_records = tuple(
        DuplicatePair(
            image_a_id=left,
            image_a_path=Path(f"data/{left}.jpg"),
            image_b_id=right,
            image_b_path=Path(f"data/{right}.jpg"),
            similarity=similarity,
            threshold=0.95,
            embedding_space_id="space-1",
        )
        for left, right, similarity in pairs
    )
    return NearDuplicateResult(1, 0.95, "space-1", 4, len(pair_records), 1, pair_records, ())


def _label_finding(image_id, *, state="review", score=0.8, source_metadata=None):
    return LabelManipulationFinding(
        image_id=image_id,
        file_path=Path(f"data/{image_id}.jpg"),
        assigned_labels=(LabelReference(4, "vehicle"),),
        nearest_neighbors=(),
        nearest_neighbor_class_distribution=(),
        same_class_neighbor_count=1,
        different_class_neighbor_count=4,
        strongest_competing_label=LabelReference(8, "aircraft"),
        similarity_to_same_class_neighbors=0.7,
        similarity_to_strongest_competing_class=0.9,
        label_inconsistency_score=score,
        configured_threshold=0.8,
        embedding_space_id="space-1",
        state=state,
        recommendation="REVIEW" if state == "review" else "NO_AUTOMATIC_ACTION",
        state_reason="synthetic test evidence",
        source_metadata=source_metadata or ContributorSourceMetadata(),
    )


def _label_result(findings=(_label_finding(2),), *, dataset_source_metadata=None):
    return LabelManipulationResult(
        schema_version=1,
        detector_name=LABEL_MANIPULATION_DETECTOR_NAME,
        detector_version="1.0.0",
        configuration=LabelManipulationConfiguration(5, 0.8, 1),
        embedding_space_id="space-1",
        total_images_analyzed=len(findings),
        suspicious_count=sum(f.state == "review" for f in findings),
        findings=tuple(findings),
        excluded_images=(),
        dataset_source_metadata=dataset_source_metadata or ContributorSourceMetadata(),
    )


def _ood_result(findings=(OODFinding(
    image_id=4,
    file_path=Path("data/4.jpg"),
    nearest_reference_image_ids=(90,),
    nearest_reference_neighbors=(),
    nearest_reference_similarity=0.4,
    configured_threshold=0.8,
    anomaly_score=0.4,
    embedding_space_id="space-1",
    reference_set_identifier="reference-v1",
    state="distribution_shift_candidate",
    interpretation="Feature representation is distant from the reference distribution.",
    recommendation="REVIEW",
),)):
    return OODDetectionResult(
        schema_version=1,
        detector_name="reference_cosine_distribution_shift",
        detector_version="1.0.0",
        threshold=0.8,
        reference_set=ReferenceSetMetadata("reference-v1", "caller", "reference", None, 1, 0),
        embedding_space_id="space-1",
        total_query_images=len(findings),
        anomalous_count=sum(f.state == "distribution_shift_candidate" for f in findings),
        findings=tuple(findings),
        unscored_queries=(),
    )


def _all_results():
    # Include an exact duplicate pair to verify deduplication.
    near = _near_result(((1, 2, 0.98), (2, 3, 0.96), (1, 2, 0.98)))
    label = _label_result((_label_finding(2), _label_finding(1, state="not_flagged")))
    ood = _ood_result()
    return near, label, ood


def _group(result, group_type, field_value):
    return next(group for group in result.groups if group.group_type == group_type and field_value in group.group_id)


def test_aggregates_by_contributor_source_batch_and_combined_provenance():
    dataset = _dataset()
    records = [
        ProvenanceMetadata(1, "team-A", "sensor-X", "batch-1"),
        ProvenanceMetadata(2, "team-A", "sensor-X", "batch-1"),
        ProvenanceMetadata(3, None, "sensor-X", None),
        ProvenanceMetadata(4, None, None, "batch-2"),
    ]
    result = ContributorRiskAggregator().aggregate(dataset, _all_results(), records)

    contributor = _group(result, "contributor", "team-A")
    source = _group(result, "source", "sensor-X")
    batch = _group(result, "batch", "batch-1")
    combined = _group(result, "combined", '"team-A"')
    assert (contributor.total_images, contributor.affected_images) == (2, 2)
    assert contributor.detector_evidence_counts.near_duplicate_findings == 2
    assert contributor.detector_evidence_counts.label_manipulation_findings == 1
    assert contributor.evidence_score == 3.5
    assert source.total_images == 3
    assert batch.total_images == 2
    assert combined.total_images == 2
    assert result.images_with_provenance == 4
    assert result.images_without_provenance == 0
    assert result.attribution_complete is True


def test_source_only_and_batch_only_do_not_infer_contributor():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1, 2)),
        (),
        [ProvenanceMetadata(1, source_id="archive-1"), ProvenanceMetadata(2, batch_id="upload-2")],
    )
    assert len([g for g in result.groups if g.group_type == "source"]) == 1
    assert len([g for g in result.groups if g.group_type == "batch"]) == 1
    assert not [g for g in result.groups if g.group_type == "contributor"]
    assert all(g.contributor_id is None for g in result.groups)


def test_combined_provenance_record_groups_all_explicit_fields():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1,)), (), [ProvenanceMetadata(1, contributor_id="c", source_id="s", batch_id="b")]
    )
    combined = next(group for group in result.groups if group.group_type == "combined")
    assert (combined.contributor_id, combined.source_id, combined.batch_id) == ("c", "s", "b")


def test_no_provenance_returns_valid_unattributed_evidence_and_status():
    result = ContributorRiskAggregator().aggregate(_dataset((1, 2)), (_near_result(((1, 2, 0.98),)),))
    assert result.aggregation_performed is False
    assert result.groups == ()
    assert result.status_message == (
        "Provenance metadata unavailable; contributor/source/batch aggregation not performed."
    )
    assert result.unattributed_summary.total_images_without_provenance == 2
    assert result.unattributed_summary.detector_evidence_counts.near_duplicate_findings == 1
    assert len(result.unattributed_summary.evidence) == 1


def test_partial_provenance_keeps_findings_for_unattributed_images():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1, 2)), (_near_result(((1, 2, 0.98),)),), [ProvenanceMetadata(1, contributor_id="c")]
    )
    assert result.aggregation_performed is True
    assert result.attribution_complete is False
    assert (result.images_with_provenance, result.images_without_provenance) == (1, 1)
    assert result.unattributed_summary.affected_images == 1
    assert result.unattributed_summary.total_findings == 1
    assert "Partial provenance" in result.status_message


def test_exact_duplicate_findings_are_deduplicated():
    pair = ((1, 2, 0.98),)
    result = ContributorRiskAggregator().aggregate(
        _dataset((1, 2)), (_near_result(pair), _near_result(pair)),
        [ProvenanceMetadata(1, contributor_id="c"), ProvenanceMetadata(2, contributor_id="c")],
    )
    group = _group(result, "contributor", "c")
    assert group.detector_evidence_counts.near_duplicate_findings == 1
    assert group.affected_images == 2


def test_duplicate_image_ids_in_dataset_are_rejected():
    dataset = _dataset((1, 1))
    with pytest.raises(ValueError, match="Duplicate Dataset image_id"):
        ContributorRiskAggregator().aggregate(dataset)


def test_conflicting_provenance_field_is_reported_and_excluded():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1,)), (), [
            ProvenanceMetadata(1, contributor_id="alice", source_id="source-X"),
            ProvenanceMetadata(1, contributor_id="bob", source_id="source-X"),
        ],
    )
    assert len(result.provenance_conflicts) == 1
    conflict = result.provenance_conflicts[0]
    assert conflict.field_name == "contributor_id"
    assert conflict.conflicting_values == ("alice", "bob")
    assert conflict.resolution == "exclude_conflicting_field_from_attribution"
    assert not [group for group in result.groups if group.group_type == "contributor"]
    assert _group(result, "source", "source-X").total_images == 1
    assert result.images_with_provenance == 1
    assert result.attribution_complete is False


def test_conflicting_duplicate_evidence_is_reported_and_excluded_from_score():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1, 2)),
        (_near_result(((1, 2, 0.98),)), _near_result(((1, 2, 0.99),))),
        [ProvenanceMetadata(1, contributor_id="c"), ProvenanceMetadata(2, contributor_id="c")],
    )
    group = _group(result, "contributor", "c")
    assert group.total_findings == 0
    assert len(result.evidence_conflicts) == 1
    assert result.evidence_conflicts[0].resolution == "exclude_conflicting_evidence_from_aggregation"


def test_empty_dataset_and_empty_findings_are_valid():
    result = ContributorRiskAggregator().aggregate(_dataset(()), ())
    assert result.schema_version == 1
    assert result.total_images == 0
    assert result.groups == ()
    assert result.unattributed_summary.total_findings == 0


def test_multiple_detector_categories_and_weighted_score():
    weights = AggregationWeights(near_duplicate=2.0, label_manipulation=3.0, distribution_shift=4.0)
    dataset = _dataset()
    result = ContributorRiskAggregator(weights).aggregate(
        dataset,
        _all_results(),
        [ProvenanceMetadata(1, contributor_id="c"), ProvenanceMetadata(2, contributor_id="c")],
    )
    group = _group(result, "contributor", "c")
    assert group.detector_evidence_counts == type(group.detector_evidence_counts)(2, 1, 0)
    assert group.evidence_score == 2 * 2.0 + 1 * 3.0
    assert group.total_findings == 3


def test_affected_image_count_deduplicates_pair_endpoints():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1, 2, 3)), (_near_result(((1, 2, 0.98), (2, 3, 0.97))),),
        [ProvenanceMetadata(2, contributor_id="only-image-2")],
    )
    group = _group(result, "contributor", "only-image-2")
    assert group.detector_evidence_counts.near_duplicate_findings == 2
    assert group.affected_images == 1


def test_configurable_review_threshold_and_default_recommendation():
    dataset = _dataset((1, 2))
    records = [ProvenanceMetadata(1, source_id="s"), ProvenanceMetadata(2, source_id="s")]
    default = ContributorRiskAggregator().aggregate(dataset, (_near_result(((1, 2, 0.98),)),), records)
    high = ContributorRiskAggregator(review_score_threshold=2.0).aggregate(
        dataset, (_near_result(((1, 2, 0.98),)),), records
    )
    assert _group(default, "source", "s").recommendation is Recommendation.REVIEW
    assert _group(high, "source", "s").recommendation is Recommendation.ACCEPT
    assert all(group.recommendation is not Recommendation.QUARANTINE for group in default.groups)


def test_deterministic_group_order_scoring_and_human_readable_explanation():
    dataset = _dataset((1, 2, 3))
    records = [
        ProvenanceMetadata(1, contributor_id="z", source_id="s2"),
        ProvenanceMetadata(2, contributor_id="a", source_id="s1"),
        ProvenanceMetadata(3, contributor_id="z", source_id="s2"),
    ]
    aggregator = ContributorRiskAggregator()
    first = aggregator.aggregate(dataset, (_near_result(((1, 3, 0.98),)),), records)
    second = aggregator.aggregate(dataset, (_near_result(((1, 3, 0.98),)),), records)
    assert first == second
    assert [group.group_id for group in first.groups] == sorted(
        [group.group_id for group in first.groups], key=lambda value: (
            0 if value.startswith("contributor") else 1 if value.startswith("source") else 3,
            value,
        )
    )
    group = _group(first, "contributor", "z")
    assert "near-duplicate" in group.human_readable_reason
    assert "2 of 2 images" in group.human_readable_reason
    assert "score 1.000" in group.human_readable_reason
    assert "REVIEW" in group.human_readable_reason


def test_result_has_versioned_provenance_and_limitations():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1,)), (), [ProvenanceMetadata(1, batch_id="b")]
    )
    assert result.schema_version == 1
    assert result.detector_name == "contributor_source_evidence_aggregation"
    assert result.detector_version == "1.0.0"
    assert result.aggregation_config.weights.near_duplicate == 1.0
    assert result.aggregation_config.weights.label_manipulation == 1.5
    assert result.aggregation_config.weights.distribution_shift == 0.5
    group = next(group for group in result.groups if group.group_type == "batch")
    assert group.batch_id == "b"
    assert group.contributor_id is None
    assert group.source_id is None
    assert any("not a calibrated probability" in item for item in result.limitations)
    assert any("QUARANTINE" in item for item in result.limitations)


def test_human_reason_lists_each_detector_category_and_affected_images():
    result = ContributorRiskAggregator().aggregate(
        _dataset((1, 2, 3, 4)), _all_results(), [ProvenanceMetadata(i, contributor_id="c") for i in (1, 2, 3, 4)]
    )
    group = _group(result, "contributor", "c")
    assert "Layer 1C near-duplicate: 2" in group.human_readable_reason
    assert "Layer 1D label-manipulation: 1" in group.human_readable_reason
    assert "Layer 1E distribution-shift: 1" in group.human_readable_reason
    assert group.affected_images == 4


def test_unknown_detector_name_is_rejected():
    with pytest.raises(UnknownDetectorError, match="unrecognized-detector"):
        ContributorRiskAggregator().aggregate(_dataset((1,)), [SimpleNamespace(detector_name="unrecognized-detector")])


def test_malformed_provenance_and_unknown_image_rejected():
    with pytest.raises(InvalidProvenanceError, match="contributor_id"):
        ContributorRiskAggregator().aggregate(
            _dataset((1,)), (), [{"image_id": 1, "contributor_id": ["bad"]}]
        )
    with pytest.raises(InvalidProvenanceError, match="absent from the Dataset"):
        ContributorRiskAggregator().aggregate(_dataset((1,)), (), [ProvenanceMetadata(2, source_id="s")])


def test_invalid_detector_scores_and_weight_configurations_rejected():
    malformed = _label_finding(1, score=float("nan"))
    with pytest.raises(InvalidDetectorEvidenceError, match="finite numeric score"):
        ContributorRiskAggregator().aggregate(_dataset((1,)), (_label_result((malformed,)),))
    with pytest.raises(ValueError, match="weight"):
        AggregationWeights(near_duplicate=float("inf"))
    with pytest.raises(ValueError, match="review_score_threshold"):
        ContributorRiskAggregator(review_score_threshold=-1)


def test_dataset_image_provenance_attributes_are_consumed_only_when_explicit():
    dataset = _dataset((1,))
    dataset.images[0].source_id = "explicit-source"
    result = ContributorRiskAggregator().aggregate(dataset)
    source = next(group for group in result.groups if group.group_type == "source")
    assert source.source_id == "explicit-source"
    assert source.contributor_id is None


def test_layer1d_provenance_is_preserved_when_it_is_not_on_dataset_images():
    result = _label_result(
        (_label_finding(1, source_metadata=ContributorSourceMetadata(contributor="from-layer-1d")),),
        dataset_source_metadata=ContributorSourceMetadata(source="dataset-source"),
    )
    summary = ContributorRiskAggregator().aggregate(_dataset((1, 2)), (result,))
    contributor_group = _group(summary, "contributor", "from-layer-1d")
    source_group = _group(summary, "source", "dataset-source")
    assert contributor_group.total_images == 1
    assert source_group.total_images == 2
    assert summary.images_with_provenance == 2


def test_duplicate_result_category_findings_count_only_once_per_image():
    finding = _label_finding(1)
    result = ContributorRiskAggregator().aggregate(
        _dataset((1,)), (_label_result((finding, finding)),), [ProvenanceMetadata(1, contributor_id="c")]
    )
    group = _group(result, "contributor", "c")
    assert group.detector_evidence_counts.label_manipulation_findings == 1
    assert group.affected_images == 1
