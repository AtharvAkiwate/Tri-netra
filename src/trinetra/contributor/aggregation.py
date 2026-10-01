"""Deterministic evidence aggregation across contributors, sources, and batches."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Union

from trinetra.detectors.label_manipulation import LABEL_MANIPULATION_SCHEMA_VERSION, LabelManipulationResult
from trinetra.detectors.near_duplicate import NEAR_DUPLICATE_SCHEMA_VERSION, NearDuplicateResult
from trinetra.detectors.ood import OODDetectionResult, OOD_SCHEMA_VERSION
from trinetra.features.models import ImageId
from trinetra.parsers.models import Dataset, DatasetImage
from trinetra.contributor.provenance import (
    PROVENANCE_FIELDS,
    InvalidProvenanceError,
    ProvenanceConflict,
    ProvenanceMetadata,
    ProvenanceValue,
    provenance_from_mapping,
)

CONTRIBUTOR_RISK_SCHEMA_VERSION = 1
CONTRIBUTOR_RISK_DETECTOR_NAME = "contributor_source_evidence_aggregation"
CONTRIBUTOR_RISK_DETECTOR_VERSION = "1.0.0"


class AggregationError(ValueError):
    """Base exception for invalid evidence aggregation inputs."""


class UnknownDetectorError(AggregationError):
    """A detector result is not supported by a Layer 1F adapter."""


class InvalidDetectorEvidenceError(AggregationError):
    """A supported detector result contains malformed evidence or scores."""


class DetectorCategory(str, Enum):
    NEAR_DUPLICATE = "near_duplicate"
    LABEL_MANIPULATION = "label_manipulation"
    DISTRIBUTION_SHIFT = "distribution_shift"


class Recommendation(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    QUARANTINE = "QUARANTINE"


@dataclass(frozen=True)
class AggregationWeights:
    """Transparent development weights; they are not calibrated probabilities."""

    near_duplicate: float = 1.0
    label_manipulation: float = 1.5
    distribution_shift: float = 0.5

    def __post_init__(self) -> None:
        for name in ("near_duplicate", "label_manipulation", "distribution_shift"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} weight must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} weight must be a finite non-negative number")
            object.__setattr__(self, name, float(value))


@dataclass(frozen=True)
class AggregationConfiguration:
    weights: AggregationWeights = AggregationWeights()
    review_score_threshold: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.weights, AggregationWeights):
            raise ValueError("weights must be AggregationWeights")
        value = self.review_score_threshold
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("review_score_threshold must be finite and non-negative")
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError("review_score_threshold must be finite and non-negative")
        object.__setattr__(self, "review_score_threshold", float(value))


@dataclass(frozen=True)
class DetectorEvidenceCounts:
    near_duplicate_findings: int = 0
    label_manipulation_findings: int = 0
    distribution_shift_findings: int = 0

    @property
    def total_findings(self) -> int:
        return self.near_duplicate_findings + self.label_manipulation_findings + self.distribution_shift_findings


@dataclass(frozen=True)
class AggregatedEvidence:
    detector_category: DetectorCategory
    evidence_id: str
    image_ids: tuple[ImageId, ...]
    summary: str
    detector_score: Optional[float]


@dataclass(frozen=True)
class DetectorEvidenceConflict:
    """Same finding identity arrived with incompatible evidence; excluded from score."""

    detector_category: DetectorCategory
    evidence_id: str
    image_ids: tuple[ImageId, ...]
    variants: tuple[str, ...]
    resolution: str = "exclude_conflicting_evidence_from_aggregation"


@dataclass(frozen=True)
class ProvenanceGroupSummary:
    group_type: str
    group_id: str
    contributor_id: Optional[ProvenanceValue]
    source_id: Optional[ProvenanceValue]
    batch_id: Optional[ProvenanceValue]
    total_images: int
    affected_images: int
    detector_evidence_counts: DetectorEvidenceCounts
    total_findings: int
    evidence_score: float
    recommendation: Recommendation
    human_readable_reason: str
    evidence: tuple[AggregatedEvidence, ...]


@dataclass(frozen=True)
class UnattributedSummary:
    total_images_without_provenance: int
    affected_images: int
    detector_evidence_counts: DetectorEvidenceCounts
    total_findings: int
    evidence_score: float
    evidence: tuple[AggregatedEvidence, ...]
    explanation: str


@dataclass(frozen=True)
class ContributorRiskSummary:
    schema_version: int
    detector_name: str
    detector_version: str
    aggregation_config: AggregationConfiguration
    total_images: int
    images_with_provenance: int
    images_without_provenance: int
    attribution_complete: bool
    aggregation_performed: bool
    status_message: str
    groups: tuple[ProvenanceGroupSummary, ...]
    unattributed_summary: UnattributedSummary
    provenance_conflicts: tuple[ProvenanceConflict, ...]
    evidence_conflicts: tuple[DetectorEvidenceConflict, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _Evidence:
    category: DetectorCategory
    key: tuple[tuple[str, str], ...]
    evidence_id: str
    image_ids: tuple[ImageId, ...]
    summary: str
    score: Optional[float]

    @property
    def signature(self) -> tuple[object, ...]:
        return (self.image_ids, self.summary, self.score)


def _image_key(image_id: ImageId) -> tuple[str, str]:
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise AggregationError(f"image_id must be an int or str, got {image_id!r}")
    return ("int" if isinstance(image_id, int) else "str", str(image_id))


def _evidence_id(key: tuple[tuple[str, str], ...]) -> str:
    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def _valid_score(value: object, label: str, minimum: float, maximum: Optional[float] = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvalidDetectorEvidenceError(f"{label} must be a finite numeric score")
    score = float(value)
    if score < minimum or (maximum is not None and score > maximum):
        range_text = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise InvalidDetectorEvidenceError(f"{label} must be {range_text}")
    return score


def _adapter_evidence(result: object) -> list[_Evidence]:
    evidence: list[_Evidence] = []
    if isinstance(result, NearDuplicateResult):
        if result.schema_version != NEAR_DUPLICATE_SCHEMA_VERSION:
            raise InvalidDetectorEvidenceError(f"Unsupported Layer 1C schema_version: {result.schema_version!r}")
        for pair in result.pairs:
            ids = (pair.image_a_id, pair.image_b_id)
            if pair.image_a_id == pair.image_b_id:
                raise InvalidDetectorEvidenceError("Layer 1C evidence cannot compare an image with itself")
            ordered_ids = tuple(sorted(ids, key=_image_key))
            key = tuple(_image_key(image_id) for image_id in ordered_ids)
            similarity = _valid_score(pair.similarity, "Layer 1C similarity", -1.0, 1.0)
            threshold = _valid_score(pair.threshold, "Layer 1C threshold", -1.0, 1.0)
            evidence.append(_Evidence(
                DetectorCategory.NEAR_DUPLICATE, key, _evidence_id(key), ordered_ids,
                f"Near-duplicate pair {ordered_ids[0]!r} ↔ {ordered_ids[1]!r}; "
                f"cosine similarity {similarity:.6f} met configured threshold {threshold:.6f}.",
                similarity,
            ))
        return evidence
    if isinstance(result, LabelManipulationResult):
        if result.schema_version != LABEL_MANIPULATION_SCHEMA_VERSION:
            raise InvalidDetectorEvidenceError(f"Unsupported Layer 1D schema_version: {result.schema_version!r}")
        for finding in result.findings:
            image_id = finding.image_id
            image_key = _image_key(image_id)
            score = _valid_score(finding.label_inconsistency_score, "Layer 1D label inconsistency score", 0.0, 1.0)
            if finding.state != "review":
                continue
            key = (image_key,)
            assigned = ", ".join(
                (label.category_name or str(label.category_id)) for label in finding.assigned_labels
            ) or "unknown assigned label"
            competing = (
                finding.strongest_competing_label.category_name
                or str(finding.strongest_competing_label.category_id)
                if finding.strongest_competing_label is not None else "unavailable"
            )
            evidence.append(_Evidence(
                DetectorCategory.LABEL_MANIPULATION, key, _evidence_id(key), (image_id,),
                f"Label-neighborhood review for image {image_id!r}: assigned [{assigned}], "
                f"strongest competing label [{competing}], score {score:.6f} "
                f"at threshold {finding.configured_threshold:.6f}.",
                score,
            ))
        return evidence
    if isinstance(result, OODDetectionResult):
        if result.schema_version != OOD_SCHEMA_VERSION:
            raise InvalidDetectorEvidenceError(f"Unsupported Layer 1E schema_version: {result.schema_version!r}")
        for finding in result.findings:
            image_id = finding.image_id
            image_key = _image_key(image_id)
            similarity = _valid_score(finding.nearest_reference_similarity, "Layer 1E cosine similarity", -1.0, 1.0)
            score = _valid_score(finding.anomaly_score, "Layer 1E anomaly score", 0.0)
            threshold = _valid_score(finding.configured_threshold, "Layer 1E threshold", -1.0, 1.0)
            if finding.state != "distribution_shift_candidate":
                continue
            key = (image_key,)
            evidence.append(_Evidence(
                DetectorCategory.DISTRIBUTION_SHIFT, key, _evidence_id(key), (image_id,),
                f"Distribution-shift candidate image {image_id!r}: nearest-reference cosine "
                f"similarity {similarity:.6f}, configured threshold {threshold:.6f}, "
                f"score {score:.6f}.",
                score,
            ))
        return evidence
    detector_name = getattr(result, "detector_name", type(result).__name__)
    raise UnknownDetectorError(f"Unsupported detector result/name: {detector_name!r}")


class ContributorRiskAggregator:
    """Aggregate existing Layer 1C/1D/1E evidence by explicit provenance."""

    def __init__(
        self,
        weights: AggregationWeights = AggregationWeights(),
        *,
        review_score_threshold: float = 0.0,
    ) -> None:
        self.configuration = AggregationConfiguration(weights, review_score_threshold)

    def aggregate(
        self,
        dataset: Dataset,
        detector_results: Iterable[object] = (),
        provenance_records: Optional[Iterable[Union[ProvenanceMetadata, Mapping[str, object]]]] = None,
    ) -> ContributorRiskSummary:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Layer 1A Dataset")
        detector_results = tuple(detector_results)
        image_by_id: dict[ImageId, DatasetImage] = {}
        for image in dataset.images:
            _image_key(image.image_id)
            if image.image_id in image_by_id:
                raise AggregationError(f"Duplicate Dataset image_id: {image.image_id!r}")
            image_by_id[image.image_id] = image

        provenance_candidates: dict[ImageId, list[ProvenanceMetadata]] = defaultdict(list)
        for image_id in image_by_id:
            for candidate in _metadata_candidates(dataset, image_by_id[image_id], image_id):
                provenance_candidates[image_id].append(candidate)
        # Layer 1D already carries optional Dataset/DatasetImage provenance in
        # its result schema. Preserve it when callers supply those results.
        for detector_result in detector_results:
            if not isinstance(detector_result, LabelManipulationResult):
                continue
            dataset_values = _detector_metadata_values(detector_result.dataset_source_metadata)
            if image_by_id and any(value is not None for value in dataset_values.values()):
                for image_id in image_by_id:
                    provenance_candidates[image_id].append(ProvenanceMetadata(image_id, **dataset_values))
            for finding in detector_result.findings:
                if finding.image_id not in image_by_id:
                    raise InvalidDetectorEvidenceError(
                        f"Layer 1D finding references image_id {finding.image_id!r} absent from Dataset"
                    )
                finding_values = _detector_metadata_values(finding.source_metadata)
                if any(value is not None for value in finding_values.values()):
                    provenance_candidates[finding.image_id].append(
                        ProvenanceMetadata(finding.image_id, **finding_values)
                    )
            for excluded in detector_result.excluded_images:
                if excluded.image_id not in image_by_id:
                    raise InvalidDetectorEvidenceError(
                        f"Layer 1D exclusion references image_id {excluded.image_id!r} absent from Dataset"
                    )
                excluded_values = _detector_metadata_values(excluded.source_metadata)
                if any(value is not None for value in excluded_values.values()):
                    provenance_candidates[excluded.image_id].append(
                        ProvenanceMetadata(excluded.image_id, **excluded_values)
                    )
        if provenance_records is not None:
            for raw_record in provenance_records:
                if isinstance(raw_record, ProvenanceMetadata):
                    record = raw_record
                elif isinstance(raw_record, Mapping):
                    record = provenance_from_mapping(raw_record)
                else:
                    raise InvalidProvenanceError(
                        "provenance_records must contain ProvenanceMetadata or mapping records"
                    )
                if record.image_id not in image_by_id:
                    raise InvalidProvenanceError(
                        f"Provenance image_id {record.image_id!r} is absent from the Dataset"
                    )
                provenance_candidates[record.image_id].append(record)

        resolved: dict[ImageId, ProvenanceMetadata] = {}
        provenance_conflicts: list[ProvenanceConflict] = []
        for image_id in sorted(image_by_id, key=_image_key):
            records = provenance_candidates.get(image_id, [])
            values: dict[str, Optional[ProvenanceValue]] = {}
            for field_name in PROVENANCE_FIELDS:
                distinct = {
                    getattr(record, field_name)
                    for record in records
                    if getattr(record, field_name) is not None
                }
                ordered_values = tuple(sorted(distinct, key=lambda value: (type(value).__name__, str(value))))
                if len(ordered_values) > 1:
                    provenance_conflicts.append(ProvenanceConflict(image_id, field_name, ordered_values))
                    values[field_name] = None
                else:
                    values[field_name] = ordered_values[0] if ordered_values else None
            resolved[image_id] = ProvenanceMetadata(image_id=image_id, **values)

        all_evidence: dict[tuple[DetectorCategory, tuple[tuple[str, str], ...]], _Evidence] = {}
        conflicting_evidence: dict[
            tuple[DetectorCategory, tuple[tuple[str, str], ...]], tuple[_Evidence, ...]
        ] = {}
        for result in detector_results:
            for item in _adapter_evidence(result):
                if any(image_id not in image_by_id for image_id in item.image_ids):
                    unknown = [image_id for image_id in item.image_ids if image_id not in image_by_id]
                    raise InvalidDetectorEvidenceError(
                        f"Detector evidence references image IDs absent from Dataset: {unknown!r}"
                    )
                key = (item.category, item.key)
                if key in conflicting_evidence:
                    variants = list(conflicting_evidence[key])
                    if all(existing.signature != item.signature for existing in variants):
                        variants.append(item)
                    conflicting_evidence[key] = tuple(variants)
                elif key in all_evidence:
                    existing = all_evidence[key]
                    if existing.signature != item.signature:
                        conflicting_evidence[key] = (existing, item)
                        del all_evidence[key]
                else:
                    all_evidence[key] = item

        evidence_conflicts = tuple(
            DetectorEvidenceConflict(
                detector_category=category,
                evidence_id=evidence_key,
                image_ids=tuple(sorted(
                    {image_id for variant in variants for image_id in variant.image_ids}, key=_image_key
                )),
                variants=tuple(sorted({variant.summary for variant in variants})),
            )
            for (category, _identity), variants in sorted(
                conflicting_evidence.items(), key=lambda item: (item[0][0].value, item[0][1])
            )
            for evidence_key in (variants[0].evidence_id,)
        )
        evidence = tuple(sorted(all_evidence.values(), key=lambda item: (item.category.value, item.key)))

        has_provenance = {
            image_id for image_id, record in resolved.items()
            if any(getattr(record, field_name) is not None for field_name in PROVENANCE_FIELDS)
        }
        without_provenance = set(image_by_id) - has_provenance
        groups = self._build_groups(resolved, evidence)
        unattributed = self._build_unattributed(without_provenance, evidence)
        conflicts_sorted = tuple(sorted(
            provenance_conflicts,
            key=lambda conflict: (_image_key(conflict.image_id), conflict.field_name),
        ))
        attribution_complete = bool(image_by_id) and not without_provenance and not conflicts_sorted
        aggregation_performed = bool(has_provenance)
        if not aggregation_performed:
            status_message = (
                "Provenance metadata unavailable; contributor/source/batch aggregation not performed."
            )
        elif not attribution_complete:
            status_message = (
                "Partial provenance: attributable groups were aggregated; unassigned evidence and conflicts are reported separately."
            )
        else:
            status_message = "Provenance available; evidence aggregation completed for all dataset images."
        return ContributorRiskSummary(
            schema_version=CONTRIBUTOR_RISK_SCHEMA_VERSION,
            detector_name=CONTRIBUTOR_RISK_DETECTOR_NAME,
            detector_version=CONTRIBUTOR_RISK_DETECTOR_VERSION,
            aggregation_config=self.configuration,
            total_images=len(image_by_id),
            images_with_provenance=len(has_provenance),
            images_without_provenance=len(without_provenance),
            attribution_complete=attribution_complete,
            aggregation_performed=aggregation_performed,
            status_message=status_message,
            groups=groups,
            unattributed_summary=unattributed,
            provenance_conflicts=conflicts_sorted,
            evidence_conflicts=evidence_conflicts,
            limitations=(
                "Evidence score is a configurable weighted count, not a calibrated probability.",
                "Evidence aggregation does not establish intent, causation, or malicious behavior.",
                "QUARANTINE is never selected automatically by this evidence-only layer.",
            ),
        )

    def _build_groups(
        self,
        provenance: dict[ImageId, ProvenanceMetadata],
        evidence: tuple[_Evidence, ...],
    ) -> tuple[ProvenanceGroupSummary, ...]:
        group_members: dict[tuple[str, object], set[ImageId]] = defaultdict(set)
        group_values: dict[tuple[str, object], tuple[Optional[ProvenanceValue], Optional[ProvenanceValue], Optional[ProvenanceValue]]] = {}
        for image_id, record in provenance.items():
            contributor, source, batch = record.contributor_id, record.source_id, record.batch_id
            for group_type, value, values in (
                ("contributor", contributor, (contributor, None, None)),
                ("source", source, (None, source, None)),
                ("batch", batch, (None, None, batch)),
            ):
                if value is not None:
                    key = (group_type, (type(value).__name__, str(value)))
                    group_members[key].add(image_id)
                    group_values[key] = values
            # A one-field record is already represented by its dedicated group;
            # combined tuples are useful when there are at least two explicit
            # identifiers, without treating missing fields as inferred identity.
            if sum(value is not None for value in (contributor, source, batch)) >= 2:
                combo = (contributor, source, batch)
                key = ("combined", combo)
                group_members[key].add(image_id)
                group_values[key] = combo

        summaries: list[ProvenanceGroupSummary] = []
        type_order = {"contributor": 0, "source": 1, "batch": 2, "combined": 3}
        for key in sorted(group_members, key=lambda item: (type_order[item[0]], repr(item[1]))):
            group_type = key[0]
            member_ids = group_members[key]
            member_evidence = tuple(
                item for item in evidence if any(image_id in member_ids for image_id in item.image_ids)
            )
            counts = _counts(member_evidence)
            score = _score(counts, self.configuration.weights)
            recommendation = self._recommend(counts, score)
            affected_ids = {
                image_id for item in member_evidence for image_id in item.image_ids if image_id in member_ids
            }
            details = "; ".join(_category_counts_text(counts)) or "No detector findings affected this group."
            summaries.append(ProvenanceGroupSummary(
                group_type=group_type,
                group_id=_format_group_id(group_type, group_values[key]),
                contributor_id=group_values[key][0],
                source_id=group_values[key][1],
                batch_id=group_values[key][2],
                total_images=len(member_ids),
                affected_images=len(affected_ids),
                detector_evidence_counts=counts,
                total_findings=counts.total_findings,
                evidence_score=score,
                recommendation=recommendation,
                human_readable_reason=(
                    f"Observed {counts.total_findings} distinct detector findings affecting "
                    f"{len(affected_ids)} of {len(member_ids)} images ({details}); "
                    f"weighted evidence score {score:.3f}. Recommendation: {recommendation.value}."
                ),
                evidence=tuple(_public_evidence(item) for item in member_evidence),
            ))
        return tuple(summaries)

    def _build_unattributed(
        self,
        without_provenance: set[ImageId],
        evidence: tuple[_Evidence, ...],
    ) -> UnattributedSummary:
        missing_evidence = tuple(
            item for item in evidence if any(image_id in without_provenance for image_id in item.image_ids)
        )
        counts = _counts(missing_evidence)
        affected_ids = {
            image_id for item in missing_evidence for image_id in item.image_ids if image_id in without_provenance
        }
        return UnattributedSummary(
            total_images_without_provenance=len(without_provenance),
            affected_images=len(affected_ids),
            detector_evidence_counts=counts,
            total_findings=counts.total_findings,
            evidence_score=_score(counts, self.configuration.weights),
            evidence=tuple(_public_evidence(item) for item in missing_evidence),
            explanation=(
                f"{counts.total_findings} distinct detector findings involve {len(affected_ids)} images "
                "without attributable contributor/source/batch metadata."
            ),
        )

    def _recommend(self, counts: DetectorEvidenceCounts, score: float) -> Recommendation:
        if counts.total_findings == 0:
            return Recommendation.ACCEPT
        if score >= self.configuration.review_score_threshold:
            return Recommendation.REVIEW
        return Recommendation.ACCEPT


def _metadata_candidates(dataset: Dataset, image: DatasetImage, image_id: ImageId) -> list[ProvenanceMetadata]:
    candidates: list[ProvenanceMetadata] = []
    for owner in (dataset, image):
        direct = {field_name: getattr(owner, field_name, None) for field_name in PROVENANCE_FIELDS}
        if any(value is not None for value in direct.values()):
            candidates.append(ProvenanceMetadata(image_id=image_id, **direct))
        metadata = getattr(owner, "metadata", None)
        if isinstance(metadata, Mapping):
            present = {field_name: metadata[field_name] for field_name in PROVENANCE_FIELDS if field_name in metadata}
            if present:
                candidates.append(ProvenanceMetadata(image_id=image_id, **present))
    return candidates


def _detector_metadata_values(metadata: object) -> dict[str, Optional[ProvenanceValue]]:
    """Map Layer 1D's explicitly named fields to Layer 1F ID fields."""
    return {
        "contributor_id": getattr(metadata, "contributor", None),
        "source_id": getattr(metadata, "source", None),
        "batch_id": getattr(metadata, "batch", None),
    }


def _counts(evidence: Iterable[_Evidence]) -> DetectorEvidenceCounts:
    counts = Counter(item.category for item in evidence)
    return DetectorEvidenceCounts(
        near_duplicate_findings=counts[DetectorCategory.NEAR_DUPLICATE],
        label_manipulation_findings=counts[DetectorCategory.LABEL_MANIPULATION],
        distribution_shift_findings=counts[DetectorCategory.DISTRIBUTION_SHIFT],
    )


def _score(counts: DetectorEvidenceCounts, weights: AggregationWeights) -> float:
    score = float(
        counts.near_duplicate_findings * weights.near_duplicate
        + counts.label_manipulation_findings * weights.label_manipulation
        + counts.distribution_shift_findings * weights.distribution_shift
    )
    if not math.isfinite(score):
        raise AggregationError("Weighted evidence score overflowed; reduce the configured weights")
    return score


def _category_counts_text(counts: DetectorEvidenceCounts) -> list[str]:
    values = (
        ("Layer 1C near-duplicate", counts.near_duplicate_findings),
        ("Layer 1D label-manipulation", counts.label_manipulation_findings),
        ("Layer 1E distribution-shift", counts.distribution_shift_findings),
    )
    return [f"{name}: {count}" for name, count in values if count]


def _public_evidence(item: _Evidence) -> AggregatedEvidence:
    return AggregatedEvidence(item.category, item.evidence_id, item.image_ids, item.summary, item.score)


def _format_group_id(
    group_type: str,
    values: tuple[Optional[ProvenanceValue], Optional[ProvenanceValue], Optional[ProvenanceValue]],
) -> str:
    contributor, source, batch = values
    if group_type == "contributor":
        return f"contributor_id={type(contributor).__name__}:{contributor}"
    if group_type == "source":
        return f"source_id={type(source).__name__}:{source}"
    if group_type == "batch":
        return f"batch_id={type(batch).__name__}:{batch}"
    return "provenance=" + json.dumps(
        {"contributor_id": contributor, "source_id": source, "batch_id": batch},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "AggregationConfiguration",
    "AggregationError",
    "AggregationWeights",
    "AggregatedEvidence",
    "CONTRIBUTOR_RISK_DETECTOR_NAME",
    "CONTRIBUTOR_RISK_DETECTOR_VERSION",
    "CONTRIBUTOR_RISK_SCHEMA_VERSION",
    "ContributorRiskAggregator",
    "ContributorRiskSummary",
    "DetectorCategory",
    "DetectorEvidenceConflict",
    "DetectorEvidenceCounts",
    "InvalidDetectorEvidenceError",
    "ProvenanceGroupSummary",
    "Recommendation",
    "UnattributedSummary",
    "UnknownDetectorError",
]
