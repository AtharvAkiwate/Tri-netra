"""Mocked offline tests for the local ResNet-50 fallback backend."""

from pathlib import Path

import numpy as np
from PIL import Image

from trinetra.features.resnet50 import ResNet50FeatureExtractor
from trinetra.features.models import FeatureBatch
from trinetra.features.preprocessing import preprocessing_metadata
from trinetra.parsers.models import Dataset, DatasetFormat, DatasetImage


class MockResNet:
    def __init__(self):
        self.evaluating = False
        self.input_shapes = []

    def eval(self):
        self.evaluating = True
        return self

    def __call__(self, batch):
        self.input_shapes.append(tuple(batch.shape))
        means = batch.mean(axis=(2, 3))
        return np.tile(means, (1, 2048 // 3 + 1))[:, :2048]


def _dataset(root: Path) -> Dataset:
    records = []
    for image_id, color in ((1, (200, 20, 10)), (2, (10, 20, 200))):
        path = root / f"{image_id}.png"
        Image.new("RGB", (31, 25), color).save(path)
        records.append(DatasetImage(image_id, path.name, 31, 25, file_path=path))
    records.append(DatasetImage("bad", "missing.png", 10, 10, file_path=root / "missing.png"))
    return Dataset("resnet-test", DatasetFormat.COCO, images=records)


def test_resnet50_mock_backend_emits_2048d_l2_vectors_and_failures(tmp_path: Path):
    weights = tmp_path / "resnet50.pth"
    weights.write_bytes(b"local mock checkpoint")
    model = MockResNet()
    extractor = ResNet50FeatureExtractor(weights, device="cpu", batch_size=2, _model=model)
    batch = extractor.extract(_dataset(tmp_path))
    assert model.evaluating
    assert model.input_shapes == [(2, 3, 224, 224)]
    assert batch.backend == "resnet50"
    assert batch.model_id == "resnet50-imagenet-avgpool-v1"
    assert batch.embedding_dim == 2048
    assert batch.device == "cpu"
    assert batch.weights_identity.startswith("sha256:")
    assert batch.embedding_space_id.startswith("sha256:")
    assert batch.input_fingerprint.startswith("sha256:")
    assert batch.cache_key.startswith("sha256:")
    assert [embedding.image_id for embedding in batch.embeddings] == [1, 2]
    assert batch.failures[0].failure_code == "image_not_found"
    np.testing.assert_allclose(np.linalg.norm(batch.as_matrix(), axis=1), 1.0, atol=1e-6)


def test_resnet_and_dinov2_embedding_spaces_are_distinct(tmp_path: Path):
    weights = tmp_path / "resnet50.pth"
    weights.write_bytes(b"resnet")
    resnet = ResNet50FeatureExtractor(weights, _model=MockResNet())
    common = dict(
        dataset_name="d", weights_identity="sha256:same", device="cpu",
        preprocessing=preprocessing_metadata(), normalization={"method": "l2"},
    )
    dino_batch = FeatureBatch(backend="dinov2", model_id="dinov2-vit-small-14", embedding_dim=384, **common)
    resnet_batch = FeatureBatch(backend=resnet.backend, model_id=resnet.model_id,
                               embedding_dim=resnet.embedding_dim, **common)
    assert resnet.embedding_dim == 2048
    assert dino_batch.embedding_space_id != resnet_batch.embedding_space_id
