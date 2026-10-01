"""Adapters from typed TRI-NETRA outputs to dashboard view models."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from trinetra.contributor.aggregation import ContributorRiskSummary
from trinetra.detectors.label_manipulation import LabelManipulationResult
from trinetra.detectors.near_duplicate import NearDuplicateResult
from trinetra.detectors.ood import OODDetectionResult
from trinetra.detectors.phash import PHashResult
from trinetra.detectors.silhouette import SilhouetteResult
from trinetra.features.models import FeatureBatch, ImageId
from trinetra.parsers.models import Dataset
from trinetra.pipeline.contracts import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardAnnotation,
    DashboardAsset,
    DashboardFinding,
    DashboardSnapshot,
    DataIntegrityRun,
)

_LAYER_LABELS = {
    "layer1c": ("1C", "cosine_near_duplicate"),
    "layer1d": ("1D", "label_consistency"),
    "layer1e": ("1E", "distribution_shift"),
    "layer1f": ("1F", "provenance_evidence_aggregation"),
    "layer1g": ("1G", "phash_duplicate"),
    "layer1h": ("1H", "silhouette_validation"),
}


def to_json_safe(value: Any) -> Any:
    """Convert evidence values to deterministic JSON-compatible structures."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Evidence cannot contain NaN or infinity")
        return value
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, Enum):
        return to_json_safe(value.value)
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: to_json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Evidence mapping keys must be strings")
        return {key: to_json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [to_json_safe(item) for item in value]
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def dashboard_snapshot_to_dict(snapshot: DashboardSnapshot) -> dict[str, Any]:
    """Serialize view data only; preserve typed source results out-of-band."""
    payload = {
        "run_id": snapshot.run_id,
        "schema_version": snapshot.schema_version,
        "mode": snapshot.mode,
        "dataset_name": snapshot.dataset_name,
        "total_assets": snapshot.total_assets,
        "assets": snapshot.assets,
        "findings": snapshot.findings,
        "active_modules": snapshot.active_modules,
        "unavailable_modules": snapshot.unavailable_modules,
        "demo_mark": snapshot.demo_mark,
        "metadata": snapshot.metadata,
    }
    return to_json_safe(payload)


def dashboard_snapshot_to_json(snapshot: DashboardSnapshot, *, indent: int = 2) -> str:
    return json.dumps(dashboard_snapshot_to_dict(snapshot), sort_keys=True, indent=indent, allow_nan=False)


def adapt_data_integrity_run(run: DataIntegrityRun) -> DashboardSnapshot:
    """Adapt Layer 1A–1H objects without running detectors or replacing results."""
    if not isinstance(run, DataIntegrityRun):
        raise TypeError("run must be a DataIntegrityRun")
    if not isinstance(run.dataset, Dataset):
        raise TypeError("run.dataset must be a Layer 1A Dataset")
    if not run.run_id.strip():
        raise ValueError("run_id must not be empty")

    assets = tuple(
        DashboardAsset(
            image_id=image.image_id,
            file_name=image.file_name,
            image_path=str(image.file_path) if image.file_path is not None else None,
            width=image.width,
            height=image.height,
            annotations=tuple(
                DashboardAnnotation(
                    annotation_id=annotation.annotation_id,
                    category_id=annotation.category_id,
                    category_name=annotation.category_name or run.dataset.categories.get(annotation.category_id),
                    bbox_xywh=annotation.bbox.as_xywh(),
                )
                for annotation in image.annotations
            ),
            provenance=to_json_safe(run.provenance_by_image.get(image.image_id, {})),
        )
        for image in run.dataset.images
    )
    results: dict[str, object] = {}
    findings: list[DashboardFinding] = []
    active = ["1A Dataset Parser"]
    if isinstance(run.feature_batch, FeatureBatch):
        results["layer1b"] = run.feature_batch
        active.append("1B Feature Extraction")
    elif run.feature_batch is not None:
        raise TypeError("feature_batch must be a Layer 1B FeatureBatch")

    supported_types = {
        "layer1c": NearDuplicateResult,
        "layer1d": LabelManipulationResult,
        "layer1e": OODDetectionResult,
        "layer1f": ContributorRiskSummary,
        "layer1g": PHashResult,
        "layer1h": SilhouetteResult,
    }
    supplied = {key: getattr(run, key) for key in supported_types}
    dataset_assets = {asset.image_id: asset for asset in assets}
    missing: list[str] = []
    for key, result_type in supported_types.items():
        result = supplied[key]
        layer, detector_type = _LAYER_LABELS[key]
        if result is None:
            missing.append(f"Layer {layer}")
            continue
        if not isinstance(result, result_type):
            raise TypeError(f"{key} must be {result_type.__name__}, got {type(result).__name__}")
        results[key] = result
        active.append(f"Layer {layer}")
        findings.extend(_adapt_result(key, result, run.run_id, detector_type, dataset_assets))

    snapshot_metadata = dict(to_json_safe(run.metadata))
    if isinstance(run.feature_batch, FeatureBatch):
        snapshot_metadata["feature_context"] = {
            "backend": run.feature_batch.backend,
            "model_id": run.feature_batch.model_id,
            "weights_identity": run.feature_batch.weights_identity,
            "embedding_dim": run.feature_batch.embedding_dim,
            "device": run.feature_batch.device,
            "embedding_space_id": run.feature_batch.embedding_space_id,
            "fallback_used": run.feature_batch.fallback_used,
            "fallback_reason": run.feature_batch.fallback_reason,
        }
    return DashboardSnapshot(
        run_id=run.run_id,
        schema_version=DASHBOARD_SCHEMA_VERSION,
        mode="pipeline",
        dataset_name=run.dataset.name,
        total_assets=len(assets),
        assets=assets,
        findings=tuple(sorted(findings, key=lambda item: (item.layer, item.finding_id))),
        active_modules=tuple(active),
        unavailable_modules=tuple(missing),
        source_results=results,
        heatmap_artifacts=run.heatmap_artifacts,
        metadata=snapshot_metadata,
    )


def _adapt_result(
    key: str,
    result: object,
    run_id: str,
    detector_type: str,
    assets: Mapping[ImageId, DashboardAsset],
) -> list[DashboardFinding]:
    layer = _LAYER_LABELS[key][0]
    reference = key
    output: list[DashboardFinding] = []

    def add(
        finding_id: str,
        finding_type: str,
        title: str,
        image_ids: tuple[ImageId, ...],
        evidence: Mapping[str, Any],
        reason: str,
        recommendation: str,
        severity: str,
        *,
        confidence: Optional[float] = None,
        image_path: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        primary_id = image_ids[0] if image_ids else None
        asset = assets.get(primary_id) if primary_id is not None else None
        output.append(DashboardFinding(
            run_id=run_id,
            schema_version=DASHBOARD_SCHEMA_VERSION,
            layer=layer,
            finding_id=finding_id,
            finding_type=finding_type,
            title=title,
            severity=severity,
            confidence=confidence,
            affected_image_id=primary_id,
            affected_image_ids=image_ids,
            image_path=image_path or (asset.image_path if asset else None),
            evidence=to_json_safe(evidence),
            reason=reason,
            recommendation=recommendation,
            source_detector=detector_type,
            original_result_reference=reference,
            metadata=to_json_safe(metadata or {}),
        ))

    if key == "layer1c":
        for pair in result.pairs:
            ids = (pair.image_a_id, pair.image_b_id)
            add(
                f"1C:{_id_token(ids[0])}:{_id_token(ids[1])}", detector_type,
                "Cosine near-duplicate candidate", ids,
                {"similarity": pair.similarity, "threshold": pair.threshold,
                 "embedding_space_id": pair.embedding_space_id,
                 "image_a_path": pair.image_a_path, "image_b_path": pair.image_b_path},
                "These feature embeddings meet the configured cosine-similarity threshold; review the pair.",
                "REVIEW", "warning", image_path=str(pair.image_a_path) if pair.image_a_path else None,
            )
    elif key == "layer1d":
        for item in result.findings:
            assigned = item.assigned_labels
            add(
                f"1D:{_id_token(item.image_id)}", detector_type,
                "Label-neighborhood consistency finding", (item.image_id,),
                {"assigned_labels": assigned, "neighbors": item.nearest_neighbors,
                 "neighbor_class_distribution": item.nearest_neighbor_class_distribution,
                 "label_inconsistency_score": item.label_inconsistency_score,
                 "configured_threshold": item.configured_threshold,
                 "embedding_space_id": item.embedding_space_id},
                item.state_reason, item.recommendation.upper(),
                "warning" if item.state == "review" else "info",
                image_path=str(item.file_path) if item.file_path else None,
                metadata={"provenance": item.source_metadata},
            )
    elif key == "layer1e":
        for item in result.findings:
            add(
                f"1E:{_id_token(item.image_id)}", detector_type,
                "Distribution-shift candidate", (item.image_id,),
                {"nearest_reference_neighbors": item.nearest_reference_neighbors,
                 "nearest_reference_similarity": item.nearest_reference_similarity,
                 "threshold": item.configured_threshold, "anomaly_score": item.anomaly_score,
                 "embedding_space_id": item.embedding_space_id,
                 "reference_set_identifier": item.reference_set_identifier},
                item.interpretation, item.recommendation.upper(),
                "warning" if item.state == "anomalous" else "info",
                image_path=str(item.file_path) if item.file_path else None,
            )
    elif key == "layer1f":
        for group in result.groups:
            ids = tuple(sorted({image_id for entry in group.evidence for image_id in entry.image_ids}, key=_image_sort_key))
            add(
                f"1F:{group.group_type}:{group.group_id}", detector_type,
                f"Provenance evidence group — {group.group_type}", ids,
                {"group_type": group.group_type, "group_id": group.group_id,
                 "contributor_id": group.contributor_id, "source_id": group.source_id,
                 "batch_id": group.batch_id, "total_images": group.total_images,
                 "affected_images": group.affected_images,
                 "detector_evidence_counts": group.detector_evidence_counts,
                 "total_findings": group.total_findings, "evidence_score": group.evidence_score,
                 "evidence": group.evidence},
                group.human_readable_reason, group.recommendation.value,
                "warning" if group.total_findings else "info",
                metadata={"attribution_complete": result.attribution_complete},
            )
    elif key == "layer1g":
        for pair in result.pairs:
            ids = (pair.image_id_a, pair.image_id_b)
            add(
                f"1G:{_id_token(ids[0])}:{_id_token(ids[1])}", detector_type,
                "Perceptually similar image candidate", ids,
                {"hash_a": pair.hash_a, "hash_b": pair.hash_b,
                 "hamming_distance": pair.hamming_distance,
                 "configured_threshold": pair.configured_threshold,
                 "file_path_a": pair.file_path_a, "file_path_b": pair.file_path_b},
                "The image hashes are within the configured Hamming-distance threshold; review as a perceptual-similarity candidate.",
                "REVIEW", "warning", image_path=str(pair.file_path_a),
            )
    elif key == "layer1h":
        for item in result.findings:
            score = item.silhouette_score
            add(
                f"1H:{_id_token(item.image_id)}", detector_type,
                "Feature/label silhouette validation", (item.image_id,),
                {"assigned_class": item.assigned_class,
                 "a_mean_intra_class_distance": item.a_mean_intra_class_distance,
                 "b_mean_nearest_competing_class_distance": item.b_mean_nearest_competing_class_distance,
                 "silhouette_score": score, "nearest_competing_class": item.nearest_competing_class,
                 "distance_metric": result.distance_metric,
                 "embedding_space_id": result.embedding_space_id},
                item.interpretation, "REVIEW" if score <= 0.0 else "NO AUTOMATIC ACTION",
                "warning" if score <= 0.0 else "info",
                image_path=str(item.file_path) if item.file_path else None,
            )
    return output


def _id_token(value: ImageId) -> str:
    return json.dumps([type(value).__name__, value], separators=(",", ":"))


def _image_sort_key(value: ImageId) -> tuple[str, str]:
    return ("int" if isinstance(value, int) else "str", str(value))


__all__ = ["adapt_data_integrity_run", "dashboard_snapshot_to_dict", "dashboard_snapshot_to_json", "to_json_safe"]
