"""Pipeline contracts and adapters shared by non-UI providers and the UI."""

from trinetra.pipeline.adapters import (
    adapt_data_integrity_run,
    dashboard_snapshot_to_dict,
    dashboard_snapshot_to_json,
    to_json_safe,
)
from trinetra.pipeline.contracts import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardAnnotation,
    DashboardAsset,
    DashboardFinding,
    DashboardSnapshot,
    DataIntegrityRun,
    HeatmapArtifact,
)

__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "DashboardAnnotation",
    "DashboardAsset",
    "DashboardFinding",
    "DashboardSnapshot",
    "DataIntegrityRun",
    "HeatmapArtifact",
    "adapt_data_integrity_run",
    "dashboard_snapshot_to_dict",
    "dashboard_snapshot_to_json",
    "to_json_safe",
]
