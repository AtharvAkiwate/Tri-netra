"""Offline tests for the DINOv2 ViT-S/14 feature extractor."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from trinetra.features.dinov2 import DINOv2FeatureExtractor
from trinetra.features.exceptions import IncompatibleWeightsError, WeightsNotFoundError
from trinetra.parsers.models import Dataset, DatasetFormat, DatasetImage


def _checkpoint(path: Path, *, hidden_size: int = 384, patch_size: int = 14) -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({
        "model_type": "dinov2", "hidden_size": hidden_size, "patch_size": patch_size,
    }), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"mock local checkpoint bytes")
    return path


class MockDinoModel:
    config = SimpleNamespace(model_type="dinov2", hidden_size=384, patch_size=14)

    def __init__(self):
        self.is_eval = False
        self.input_shapes = []

    def eval(self):
        self.is_eval = True
        return self

    def __call__(self, pixel_values):
        # Deterministic feature function, independent per row and batch shape.
        self.input_shapes.append(tuple(pixel_values.shape))
        means = pixel_values.mean(axis=(2, 3))
        base = np.tile(means, (1, 128)).astype(np.float32)
        tokens = np.stack([base, base * 0.5], axis=1)
        return SimpleNamespace(last_hidden_state=tokens)


def _dataset(tmp_path: Path) -> Dataset:
    images = []
    for image_id, color in ((1, (240, 20, 10)), (2, (10, 20, 240)), (3, (30, 180, 50))):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (37, 23), color).save(path)
        images.append(DatasetImage(image_id, path.name, 37, 23, file_path=path))
    missing = DatasetImage("missing", "missing.png", 20, 20, file_path=tmp_path / "missing.png")
    return Dataset("demo", DatasetFormat.COCO, images=[*images, missing])


def test_dinov2_cpu_preprocessing_and_structured_failures(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "weights")
    model = MockDinoModel()
    extractor = DINOv2FeatureExtractor(checkpoint, device="cpu", batch_size=2, _model=model)
    batch = extractor.extract(_dataset(tmp_path))

    assert model.is_eval
    assert all(shape[1:] == (3, 224, 224) for shape in model.input_shapes)
    assert batch.device == "cpu"
    assert batch.backend == "dinov2"
    assert batch.model_id == "dinov2-vit-small-14"
    assert batch.embedding_dim == 384
    assert batch.weights_identity.startswith("sha256:")
    assert batch.embedding_space_id.startswith("sha256:")
    assert batch.preprocessing["resize"] == {"shortest_edge": 256, "resample": "bicubic"}
    assert batch.preprocessing["center_crop"] == {
        "height": 224, "width": 224, "rounding": "nearest-even"
    }
    assert batch.preprocessing["normalization_mean"] == [0.485, 0.456, 0.406]
    assert batch.normalization["method"] == "l2"
    assert [item.image_id for item in batch.embeddings] == [1, 2, 3]
    assert [failure.image_id for failure in batch.failures] == ["missing"]
    assert batch.failures[0].failure_code == "image_not_found"
    np.testing.assert_allclose(np.linalg.norm(batch.as_matrix(), axis=1), 1.0, atol=1e-6)


def test_dinov2_batch_sizes_produce_equivalent_vectors(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "weights")
    dataset = _dataset(tmp_path)
    batch_one = DINOv2FeatureExtractor(checkpoint, batch_size=1, _model=MockDinoModel()).extract(dataset)
    batch_many = DINOv2FeatureExtractor(checkpoint, batch_size=3, _model=MockDinoModel()).extract(dataset)
    np.testing.assert_allclose(batch_one.as_matrix(), batch_many.as_matrix(), rtol=1e-6, atol=1e-6)


def test_dinov2_auto_uses_cpu_without_cuda_runtime(tmp_path: Path):
    extractor = DINOv2FeatureExtractor(_checkpoint(tmp_path / "weights"), device="auto", _model=MockDinoModel())
    assert extractor.device == "cpu"


def test_dinov2_cpu_uses_eval_and_no_grad_runtime(tmp_path: Path):
    class Tensor:
        def __init__(self, array):
            self.array = array

        def to(self, device):
            assert device == "cpu"
            return self

    class Torch:
        class cuda:
            @staticmethod
            def is_available():
                return False

        def __init__(self):
            self.no_grad_entered = False

        @staticmethod
        def from_numpy(array):
            return Tensor(array)

        def no_grad(self):
            runtime = self

            class Context:
                def __enter__(self):
                    runtime.no_grad_entered = True

                def __exit__(self, *args):
                    return False

            return Context()

    class RuntimeModel(MockDinoModel):
        def to(self, device):
            assert device == "cpu"
            return self

        def __call__(self, *, pixel_values):
            return super().__call__(pixel_values.array)

    torch_runtime = Torch()
    extractor = DINOv2FeatureExtractor(
        _checkpoint(tmp_path / "weights"), device="cpu", _model=RuntimeModel(), _torch=torch_runtime
    )
    batch = extractor.extract(_dataset(tmp_path))
    assert torch_runtime.no_grad_entered
    assert batch.device == "cpu"


def test_dinov2_missing_weights_and_incompatible_config(tmp_path: Path):
    with pytest.raises(WeightsNotFoundError, match="directory not found"):
        DINOv2FeatureExtractor(tmp_path / "absent", _model=MockDinoModel())
    incompatible = _checkpoint(tmp_path / "wrong", hidden_size=768)
    with pytest.raises(IncompatibleWeightsError, match="ViT-S/14"):
        DINOv2FeatureExtractor(incompatible, _model=MockDinoModel())


def test_dinov2_invalid_batch_size(tmp_path: Path):
    with pytest.raises(ValueError, match="batch_size"):
        DINOv2FeatureExtractor(_checkpoint(tmp_path / "weights"), batch_size=0, _model=MockDinoModel())


def test_dinov2_optional_local_checkpoint_integration(tmp_path: Path):
    weights_path = os.environ.get("TRINETRA_DINOV2_WEIGHTS")
    if not weights_path:
        pytest.skip("Set TRINETRA_DINOV2_WEIGHTS to a local DINOv2 ViT-S/14 Transformers checkpoint")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    image_path = tmp_path / "integration.png"
    Image.new("RGB", (56, 56), (120, 80, 40)).save(image_path)
    dataset = Dataset("integration", DatasetFormat.COCO, images=[
        DatasetImage("one", image_path.name, 56, 56, file_path=image_path)
    ])
    batch = DINOv2FeatureExtractor(weights_path, device="cpu", batch_size=1).extract(dataset)
    assert batch.embedding_count == 1
    assert batch.embeddings[0].vector.shape == (384,)
