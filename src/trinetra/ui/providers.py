"""Dashboard data provider interfaces and deterministic demo/pipeline providers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

from PIL import Image, ImageDraw

from trinetra.pipeline.adapters import adapt_data_integrity_run
from trinetra.pipeline.contracts import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardAnnotation,
    DashboardAsset,
    DashboardFinding,
    DashboardSnapshot,
    DataIntegrityRun,
    HeatmapArtifact,
)

DEMO_HEATMAP_LABEL = "DEMO HEATMAP — NOT MODEL OUTPUT"
_DEMO_RUN_ID = "demo-run-2026-001"


class DashboardDataProvider(Protocol):
    """Provider contract consumed by the Streamlit presentation layer."""

    def load(self) -> DashboardSnapshot: ...


@dataclass(frozen=True)
class DemoProvider:
    """Return stable synthetic imagery and findings, visibly marked as demo data."""

    def load(self) -> DashboardSnapshot:
        assets: list[DashboardAsset] = []
        image_artifacts: dict[str, bytes] = {}
        for index, image_id in enumerate(("IMG-0441", "IMG-0442", "IMG-0443")):
            image = _demo_image(index)
            image_artifacts[image_id] = _png_bytes(image)
            assets.append(DashboardAsset(
                image_id=image_id,
                file_name=f"sample-asset-{index + 1}.png",
                image_path=None,
                width=image.width,
                height=image.height,
                annotations=(
                    DashboardAnnotation(
                        annotation_id=f"demo-ann-{index + 1}",
                        category_id=3 if index == 0 else 7,
                        category_name="vehicle" if index == 0 else "aircraft",
                        bbox_xywh=(180.0 + index * 18, 112.0 + index * 9, 225.0, 136.0),
                    ),
                ),
                provenance={"source_id": "DEMO-SOURCE-A", "batch_id": "DEMO-BATCH-07"},
            ))

        findings = (
            DashboardFinding(
                run_id=_DEMO_RUN_ID, schema_version=DASHBOARD_SCHEMA_VERSION,
                layer="1C", finding_id="demo-1c-pair-01", finding_type="cosine_near_duplicate",
                title="Near-duplicate pair candidate", severity="warning", confidence=0.91,
                affected_image_id="IMG-0441", affected_image_ids=("IMG-0441", "IMG-0442"),
                image_path=None, evidence={"cosine_similarity": 0.976, "threshold": 0.95,
                                           "comparison_asset": "IMG-0442"},
                reason="The synthetic feature vectors meet the configured similarity threshold.",
                recommendation="REVIEW", source_detector="demo.near_duplicate",
                original_result_reference="demo:layer1c", metadata={"demo": True},
            ),
            DashboardFinding(
                run_id=_DEMO_RUN_ID, schema_version=DASHBOARD_SCHEMA_VERSION,
                layer="1D", finding_id="demo-1d-label-02", finding_type="label_consistency",
                title="Label-neighborhood review candidate", severity="warning", confidence=0.78,
                affected_image_id="IMG-0442", affected_image_ids=("IMG-0442",),
                image_path=None, evidence={"label_inconsistency_score": 0.67,
                                           "nearest_neighbor_classes": {"vehicle": 4, "aircraft": 1}},
                reason="The synthetic label neighborhood is inconsistent with the assigned class.",
                recommendation="REVIEW", source_detector="demo.label_consistency",
                original_result_reference="demo:layer1d", metadata={"demo": True},
            ),
            DashboardFinding(
                run_id=_DEMO_RUN_ID, schema_version=DASHBOARD_SCHEMA_VERSION,
                layer="1E", finding_id="demo-1e-shift-03", finding_type="distribution_shift",
                title="Reference distribution-shift candidate", severity="info", confidence=0.62,
                affected_image_id="IMG-0443", affected_image_ids=("IMG-0443",),
                image_path=None, evidence={"nearest_reference_similarity": 0.74, "threshold": 0.80,
                                           "reference_set": "DEMO-REFERENCE-01"},
                reason="The synthetic feature representation is distant from the sample reference set.",
                recommendation="REVIEW", source_detector="demo.ood",
                original_result_reference="demo:layer1e", metadata={"demo": True},
            ),
            DashboardFinding(
                run_id=_DEMO_RUN_ID, schema_version=DASHBOARD_SCHEMA_VERSION,
                layer="1G", finding_id="demo-1g-phash-04", finding_type="phash_duplicate",
                title="Perceptually similar image candidate", severity="warning", confidence=0.88,
                affected_image_id="IMG-0441", affected_image_ids=("IMG-0441", "IMG-0442"),
                image_path=None, evidence={"hamming_distance": 4, "threshold": 8,
                                           "comparison_asset": "IMG-0442"},
                reason="The sample hashes are within the development Hamming-distance threshold.",
                recommendation="QUARANTINE", source_detector="demo.phash",
                original_result_reference="demo:layer1g", metadata={"demo": True},
            ),
        )
        heatmap_image = _demo_heatmap(assets[0].width, assets[0].height)
        return DashboardSnapshot(
            run_id=_DEMO_RUN_ID,
            schema_version=DASHBOARD_SCHEMA_VERSION,
            mode="demo",
            dataset_name="DEMO • Coastal Operations Sample",
            total_assets=len(assets),
            assets=tuple(assets),
            findings=findings,
            active_modules=("1A Dataset Parser", "1B Features (sample)", "1C Cosine", "1D Labels", "1E OOD", "1G pHash"),
            unavailable_modules=("1F Provenance Aggregation", "1H Silhouette"),
            source_results={},
            image_artifacts=image_artifacts,
            heatmap_artifacts={
                assets[0].image_id: HeatmapArtifact(
                    image_bytes=_png_bytes(heatmap_image), is_demo=True, label=DEMO_HEATMAP_LABEL
                )
            },
            demo_mark="DEMO / SAMPLE DATA",
            metadata={"provider": "deterministic-demo", "synthetic": True},
        )


@dataclass(frozen=True)
class PipelineProvider:
    """Adapt a real typed pipeline run, or return an honest not-connected state."""

    run: DataIntegrityRun | None = None

    def load(self) -> DashboardSnapshot:
        if self.run is not None:
            return adapt_data_integrity_run(self.run)
        return DashboardSnapshot(
            run_id="pipeline-not-connected",
            schema_version=DASHBOARD_SCHEMA_VERSION,
            mode="pipeline",
            dataset_name="No pipeline run supplied",
            total_assets=0,
            assets=(),
            findings=(),
            active_modules=(),
            unavailable_modules=("Layer 1A–1H",),
            metadata={"status": "Waiting for a DataIntegrityRun from the pipeline provider"},
        )


def _demo_image(index: int) -> Image.Image:
    width, height = 720, 420
    image = Image.new("RGB", (width, height), (19, 35, 43))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 275, width, height), fill=(28, 52, 52))
    for y in range(285, height, 28):
        draw.line((0, y, width, y - 8), fill=(42, 72, 67), width=1)
    for x in range(24, width, 45):
        draw.line((x, 276, x - 18, height), fill=(39, 65, 61), width=1)
    # Reproducible stylized target and scene markings; no random generation.
    offset = index * 18
    draw.rounded_rectangle((220 + offset, 175 + offset // 3, 395 + offset, 245 + offset // 3), radius=10, fill=(117, 133, 111), outline=(174, 190, 162), width=2)
    draw.polygon(((265 + offset, 173), (304 + offset, 143), (355 + offset, 174)), fill=(142, 158, 132))
    draw.rectangle((442, 102, 512, 150), fill=(61, 79, 83), outline=(119, 154, 155), width=2)
    draw.text((24, 22), f"SAMPLE SENSOR FRAME  /  SECTOR {chr(65 + index)}", fill=(112, 196, 194))
    draw.text((24, 388), "SYNTHETIC IMAGE • NOT OPERATIONAL DATA", fill=(140, 158, 150))
    return image


def _demo_heatmap(width: int, height: int) -> Image.Image:
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((242, 125, 444, 302), fill=(238, 53, 45, 118))
    draw.ellipse((283, 155, 402, 264), fill=(255, 38, 28, 92))
    return overlay


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False)
    return stream.getvalue()


__all__ = ["DashboardDataProvider", "DemoProvider", "PipelineProvider", "DEMO_HEATMAP_LABEL"]
