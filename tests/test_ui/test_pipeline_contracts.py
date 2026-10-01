import json
from pathlib import Path

import numpy as np
import pytest

from trinetra.contributor import ContributorRiskAggregator, ProvenanceMetadata
from trinetra.detectors import (
    LabelManipulationDetector,
    NearDuplicateDetector,
    OODDetector,
    PHashDuplicateDetector,
    SilhouetteDetector,
)
from trinetra.features.models import FeatureBatch, ImageEmbedding
from trinetra.parsers.models import Annotation, BoundingBox, Dataset, DatasetFormat, DatasetImage
from trinetra.pipeline.adapters import (
    adapt_data_integrity_run,
    dashboard_snapshot_to_dict,
    dashboard_snapshot_to_json,
    to_json_safe,
)
from trinetra.pipeline.contracts import (
    DASHBOARD_SCHEMA_VERSION,
    DashboardFinding,
    DashboardSnapshot,
    DataIntegrityRun,
    HeatmapArtifact,
)
from trinetra.ui.components.image_view import (
    PIPELINE_HEATMAP_MISSING,
    compose_asset_image,
    render_image_view,
    scale_bbox_xywh,
)
from trinetra.ui.providers import DEMO_HEATMAP_LABEL, DemoProvider, PipelineProvider
from PIL import Image


def _batch(dataset, vectors, ids=None):
    ids = ids or [image.image_id for image in dataset.images]
    lookup = {image.image_id: image for image in dataset.images}
    return FeatureBatch(
        dataset_name=dataset.name,
        backend="mock",
        model_id="synthetic-v1",
        weights_identity="sha256:synthetic",
        embedding_dim=2,
        device="cpu",
        embeddings=[
            ImageEmbedding(image_id, lookup[image_id].file_name, np.asarray(vector, dtype=np.float32), lookup[image_id].file_path)
            for image_id, vector in zip(ids, vectors, strict=True)
        ],
        preprocessing={"method": "synthetic"},
        normalization={"method": "l2"},
    )


def _pipeline_inputs(tmp_path):
    images = []
    for image_id, category_id, vector_color in (
        (1, 10, "white"), (2, 10, "white"), (3, 20, "black"), (4, 20, "black")
    ):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (64, 48), vector_color).save(path)
        images.append(DatasetImage(
            image_id, path.name, 64, 48,
            [Annotation(f"ann-{image_id}", category_id, BoundingBox(4, 6, 20, 15), f"class-{category_id}")],
            file_path=path,
        ))
    dataset = Dataset("ui-test", DatasetFormat.COCO, images, {10: "vehicle", 20: "aircraft"})
    batch = _batch(dataset, [[1, 0], [1, 0], [0, 1], [0, 1]])
    near = NearDuplicateDetector().detect(batch)
    labels = LabelManipulationDetector(k=1).detect(dataset, batch)
    reference = FeatureBatch(
        dataset_name="reference", backend=batch.backend, model_id=batch.model_id,
        weights_identity=batch.weights_identity, embedding_dim=2, device="cpu",
        embeddings=[
            ImageEmbedding(90, "90.png", np.array([1, 0], dtype=np.float32)),
            ImageEmbedding(91, "91.png", np.array([0, 1], dtype=np.float32)),
        ], preprocessing=batch.preprocessing, normalization=batch.normalization,
    )
    query = FeatureBatch(
        dataset_name="query", backend=batch.backend, model_id=batch.model_id,
        weights_identity=batch.weights_identity, embedding_dim=2, device="cpu",
        embeddings=[ImageEmbedding(1, "1.png", np.array([-1, 0], dtype=np.float32), images[0].file_path)],
        preprocessing=batch.preprocessing, normalization=batch.normalization,
    )
    ood = OODDetector().detect(reference, query, reference_set_id="ui-ref")
    provenance = [ProvenanceMetadata(i, contributor_id="unit-7", source_id="sensor-3", batch_id="batch-2") for i in range(1, 5)]
    contributor = ContributorRiskAggregator().aggregate(dataset, [near, labels, ood], provenance)
    phash = PHashDuplicateDetector().detect(dataset)
    silhouette = SilhouetteDetector().analyze(dataset, batch)
    overlay = Image.new("RGBA", (64, 48), (245, 20, 10, 72))
    overlay_buffer = __import__("io").BytesIO()
    overlay.save(overlay_buffer, format="PNG")
    run = DataIntegrityRun(
        run_id="run-ui-real-01", dataset=dataset, feature_batch=batch,
        layer1c=near, layer1d=labels, layer1e=ood, layer1f=contributor,
        layer1g=phash, layer1h=silhouette,
        provenance_by_image={1: {"contributor_id": "unit-7"}},
        heatmap_artifacts={1: HeatmapArtifact(image_bytes=overlay_buffer.getvalue())},
        metadata={"pipeline": "unit-test"},
    )
    return run, {"near": near, "labels": labels, "ood": ood, "contributor": contributor, "phash": phash, "silhouette": silhouette}


def test_dashboard_view_model_validates_schema_and_creation():
    finding = DashboardFinding(
        run_id="r1", schema_version=DASHBOARD_SCHEMA_VERSION, layer="1C",
        finding_id="f1", finding_type="duplicate", title="Candidate", severity="warning",
        confidence=None, affected_image_id=1, affected_image_ids=(1, 2), image_path=None,
        evidence={"similarity": 0.97}, reason="Review the candidate pair.", recommendation="REVIEW",
        source_detector="test", original_result_reference="layer1c",
    )
    snapshot = DashboardSnapshot("r1", 1, "pipeline", "data", 2, (), (finding,), (), (), source_results={"layer1c": object()})
    assert snapshot.findings[0] is finding
    assert snapshot.source_results["layer1c"] is not None
    with pytest.raises(ValueError, match="confidence"):
        DashboardFinding("r", 1, "1C", "f", "t", "title", "warning", 1.2, None, (), None, {}, "reason", "REVIEW", "d", "ref")


def test_demo_provider_is_deterministic_and_heatmap_is_marked_demo():
    first, second = DemoProvider().load(), DemoProvider().load()
    assert first == second
    assert first.mode == "demo"
    assert first.demo_mark == "DEMO / SAMPLE DATA"
    assert first.image_artifacts == second.image_artifacts
    artifact = first.heatmap_artifacts["IMG-0441"]
    assert artifact.is_demo is True
    assert artifact.label == DEMO_HEATMAP_LABEL
    assert len(first.findings) > 0
    assert any(item.confidence is not None for item in first.findings)


def test_real_data_adapters_cover_layers_1c_through_1h_and_keep_results(tmp_path):
    run, results = _pipeline_inputs(tmp_path)
    snapshot = adapt_data_integrity_run(run)
    assert snapshot.mode == "pipeline"
    assert snapshot.run_id == "run-ui-real-01"
    assert {item.layer for item in snapshot.findings} == {"1C", "1D", "1E", "1F", "1G", "1H"}
    assert snapshot.source_results["layer1c"] is results["near"]
    assert snapshot.source_results["layer1d"] is results["labels"]
    assert snapshot.source_results["layer1e"] is results["ood"]
    assert snapshot.source_results["layer1f"] is results["contributor"]
    assert snapshot.source_results["layer1g"] is results["phash"]
    assert snapshot.source_results["layer1h"] is results["silhouette"]
    assert snapshot.assets[0].annotations[0].bbox_xywh == (4.0, 6.0, 20.0, 15.0)
    assert snapshot.assets[0].image_path == str(run.dataset.images[0].file_path)
    assert snapshot.assets[0].provenance["contributor_id"] == "unit-7"
    assert snapshot.metadata["feature_context"]["embedding_space_id"] == run.feature_batch.embedding_space_id
    assert all(item.run_id == run.run_id and item.confidence is None for item in snapshot.findings)
    assert snapshot.heatmap_artifacts[1] is run.heatmap_artifacts[1]


def test_pipeline_provider_gracefully_handles_missing_results():
    empty = Dataset("empty-pipeline", DatasetFormat.COCO, [])
    snapshot = PipelineProvider(DataIntegrityRun("r-empty", empty)).load()
    assert snapshot.mode == "pipeline"
    assert snapshot.findings == ()
    assert "Layer 1H" in snapshot.unavailable_modules
    assert snapshot.total_assets == 0
    assert PipelineProvider().load().run_id == "pipeline-not-connected"


def test_missing_image_and_missing_pipeline_heatmap_are_graceful():
    class StubStreamlit:
        def __init__(self):
            self.messages = []
            self.images = []
        def info(self, message): self.messages.append(message)
        def warning(self, message): self.messages.append(message)
        def caption(self, message): self.messages.append(message)
        def image(self, image, **kwargs): self.images.append(image)

    from trinetra.pipeline.contracts import DashboardAsset
    stub = StubStreamlit()
    missing = DashboardAsset(4, "missing.png", None, 20, 20)
    assert not render_image_view(stub, missing, pipeline_mode=True)
    assert "Image unavailable" in stub.messages[-1]

    available = DashboardAsset(5, "memory.png", None, 10, 10)
    import io
    buffer = io.BytesIO()
    Image.new("RGB", (10, 10), "navy").save(buffer, format="PNG")
    assert render_image_view(stub, available, image_bytes=buffer.getvalue(), pipeline_mode=True)
    assert PIPELINE_HEATMAP_MISSING in stub.messages
    assert len(stub.images) == 1

    from io import BytesIO
    mismatched = BytesIO()
    Image.new("RGBA", (3, 3), (255, 0, 0, 100)).save(mismatched, format="PNG")
    assert render_image_view(
        stub, available, image_bytes=buffer.getvalue(),
        heatmap=HeatmapArtifact(image_bytes=mismatched.getvalue()), pipeline_mode=True,
    )
    assert stub.messages[-1] == PIPELINE_HEATMAP_MISSING


def test_bounding_box_coordinates_scale_from_absolute_xywh():
    assert scale_bbox_xywh((10, 20, 30, 40), (100, 100), (200, 50)) == (20, 10, 60, 20)
    from trinetra.pipeline.contracts import DashboardAnnotation, DashboardAsset
    asset = DashboardAsset(1, "x.png", None, 100, 100, (DashboardAnnotation("a", 2, "car", (10, 20, 30, 40)),))
    rendered = compose_asset_image(asset, Image.new("RGB", (100, 100)), target_size=(200, 50))
    assert rendered.size == (200, 50)


def test_serialization_is_json_safe_and_excludes_binary_and_typed_result_objects():
    snapshot = DemoProvider().load()
    payload = dashboard_snapshot_to_dict(snapshot)
    assert "image_artifacts" not in payload
    assert "source_results" not in payload
    encoded = dashboard_snapshot_to_json(snapshot)
    assert json.loads(encoded)["schema_version"] == DASHBOARD_SCHEMA_VERSION
    assert to_json_safe({"path": Path("a/b"), "score": np.float32(0.75)}) == {"path": "a\\b", "score": pytest.approx(0.75)}
    with pytest.raises(ValueError, match="NaN"):
        to_json_safe(float("nan"))
