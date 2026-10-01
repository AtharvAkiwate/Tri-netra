"""Run-level DINOv2 primary / ResNet-50 fallback policy tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trinetra.features.exceptions import FeatureExtractionError, WeightsNotFoundError
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure
from trinetra.features.preprocessing import preprocessing_metadata
from trinetra.features.runner import FeatureExtractionRunner
from trinetra.parsers.models import Dataset, DatasetFormat, DatasetImage


def _dataset(tmp_path: Path) -> Dataset:
    records = []
    for image_id in (10, 11):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (18, 18), (image_id, 50, 100)).save(path)
        records.append(DatasetImage(image_id, path.name, 18, 18, file_path=path))
    return Dataset("runner", DatasetFormat.COCO, images=records)


def _batch(dataset: Dataset, backend: str, dim: int, ids: list[int], **provenance) -> FeatureBatch:
    embeddings = [
        ImageEmbedding(image_id, f"{image_id}.png", np.ones(dim, dtype=np.float32))
        for image_id in ids
    ]
    return FeatureBatch(
        dataset_name=dataset.name,
        backend=backend,
        model_id="dinov2-vit-small-14" if backend == "dinov2" else "resnet50-imagenet-avgpool-v1",
        embedding_dim=dim,
        device="cpu",
        embeddings=embeddings,
        weights_identity=f"sha256:{backend}",
        preprocessing=preprocessing_metadata(),
        normalization={"method": "l2"},
        input_fingerprint="sha256:runner-input",
        **provenance,
    )


class FakeExtractor:
    def __init__(self, batch: FeatureBatch, seen: list):
        self.batch = batch
        self.seen = seen

    def extract(self, dataset: Dataset) -> FeatureBatch:
        self.seen.append(tuple(image.image_id for image in dataset.images))
        return self.batch


def test_fallback_is_run_level_and_never_mixes_embeddings(tmp_path: Path):
    dataset = _dataset(tmp_path)
    primary_calls, fallback_calls = [], []
    partial_primary = _batch(dataset, "dinov2", 384, [10])
    partial_primary.failures.append(ImageFailure(11, dataset.images[1].file_path,
                                                 "model_inference_error", "inference failed"))
    full_fallback = _batch(dataset, "resnet50", 2048, [10, 11])
    runner = FeatureExtractionRunner(
        lambda: FakeExtractor(partial_primary, primary_calls),
        lambda: FakeExtractor(full_fallback, fallback_calls),
        allow_fallback=True,
    )
    result = runner.extract(dataset)

    assert primary_calls == [(10, 11)]
    assert fallback_calls == [(10, 11)]
    assert [item.image_id for item in result.embeddings] == [10, 11]
    assert all(item.vector.size == 2048 for item in result.embeddings)
    assert result.backend == result.actual_extractor == "resnet50"
    assert result.requested_extractor == "dinov2"
    assert result.fallback_used is True
    assert "model_inference_error" in result.fallback_reason
    assert result.embedding_space_id == full_fallback.embedding_space_id
    assert result.schema_version == 3


def test_fallback_disabled_fails_clearly_on_primary_initialization(tmp_path: Path):
    dataset = _dataset(tmp_path)

    def missing_primary():
        raise WeightsNotFoundError("local DINO weights absent")

    runner = FeatureExtractionRunner(missing_primary, lambda: pytest.fail("fallback must not run"))
    with pytest.raises(FeatureExtractionError, match="fallback is disabled.*local DINO weights absent"):
        runner.extract(dataset)


def test_explicit_fallback_records_provenance_and_schema_round_trip(tmp_path: Path):
    dataset = _dataset(tmp_path)
    primary_calls, fallback_calls = [], []

    def unavailable():
        raise WeightsNotFoundError("DINO checkpoint missing")

    fallback_batch = _batch(dataset, "resnet50", 2048, [10, 11])
    result = FeatureExtractionRunner(
        unavailable, lambda: FakeExtractor(fallback_batch, fallback_calls), allow_fallback=True
    ).extract(dataset)
    assert result.requested_extractor == "dinov2"
    assert result.actual_extractor == "resnet50"
    assert result.fallback_used
    assert result.fallback_reason == "WeightsNotFoundError: DINO checkpoint missing"
    assert result.cache_key != fallback_batch.cache_key
    assert not primary_calls
