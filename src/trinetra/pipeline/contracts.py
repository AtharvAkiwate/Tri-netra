"""Versioned presentation contracts and pipeline run inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from trinetra.features.models import ImageId
from trinetra.parsers.models import Dataset

DASHBOARD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DashboardAnnotation:
    annotation_id: int | str
    category_id: int
    category_name: Optional[str]
    # Absolute pixel XYWH, copied from Layer 1A without transformation.
    bbox_xywh: tuple[float, float, float, float]


@dataclass(frozen=True)
class DashboardAsset:
    image_id: ImageId
    file_name: str
    image_path: Optional[str]
    width: int
    height: int
    annotations: tuple[DashboardAnnotation, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DashboardFinding:
    run_id: str
    schema_version: int
    layer: str
    finding_id: str
    finding_type: str
    title: str
    severity: str
    confidence: Optional[float]
    affected_image_id: Optional[ImageId]
    affected_image_ids: tuple[ImageId, ...]
    image_path: Optional[str]
    evidence: Mapping[str, Any]
    reason: str
    recommendation: str
    source_detector: str
    original_result_reference: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "layer", "finding_id", "finding_type", "title", "severity", "reason", "source_detector", "original_result_reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != DASHBOARD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dashboard finding schema_version: {self.schema_version!r}")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("confidence must be None or finite within [0, 1]")


@dataclass(frozen=True)
class HeatmapArtifact:
    """Aligned visualization artifact; demo artifacts must carry an explicit label."""

    image_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    is_demo: bool = False
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.image_bytes is None) == (self.image_path is None):
            raise ValueError("HeatmapArtifact requires exactly one of image_bytes or image_path")
        if self.is_demo and self.label != "DEMO HEATMAP — NOT MODEL OUTPUT":
            raise ValueError("Demo heatmap artifacts must carry the DEMO HEATMAP label")


@dataclass(frozen=True)
class DashboardSnapshot:
    """UI-ready snapshot while retaining original detector result objects."""

    run_id: str
    schema_version: int
    mode: str
    dataset_name: str
    total_assets: int
    assets: tuple[DashboardAsset, ...]
    findings: tuple[DashboardFinding, ...]
    active_modules: tuple[str, ...]
    unavailable_modules: tuple[str, ...]
    source_results: Mapping[str, object] = field(default_factory=dict)
    image_artifacts: Mapping[ImageId, bytes] = field(default_factory=dict, repr=False)
    heatmap_artifacts: Mapping[ImageId, HeatmapArtifact] = field(default_factory=dict, repr=False)
    demo_mark: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != DASHBOARD_SCHEMA_VERSION:
            raise ValueError(f"Unsupported dashboard schema_version: {self.schema_version!r}")
        if self.mode not in {"demo", "pipeline"}:
            raise ValueError("mode must be 'demo' or 'pipeline'")
        if self.mode == "demo" and self.demo_mark != "DEMO / SAMPLE DATA":
            raise ValueError("Demo snapshots must be visibly marked")


@dataclass(frozen=True)
class DataIntegrityRun:
    """Typed handoff from an external pipeline runner into the dashboard adapter."""

    run_id: str
    dataset: Dataset
    feature_batch: object | None = None
    layer1c: object | None = None
    layer1d: object | None = None
    layer1e: object | None = None
    layer1f: object | None = None
    layer1g: object | None = None
    layer1h: object | None = None
    provenance_by_image: Mapping[ImageId, Mapping[str, Any]] = field(default_factory=dict)
    heatmap_artifacts: Mapping[ImageId, HeatmapArtifact] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "DashboardAnnotation",
    "DashboardAsset",
    "DashboardFinding",
    "DashboardSnapshot",
    "DataIntegrityRun",
    "HeatmapArtifact",
]
