"""Tests for dummy backbone extraction over Layer 1A Dataset records."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trinetra.features.backend import FeatureExtractor
from trinetra.features.dummy import DummyFeatureExtractor
from trinetra.features.exceptions import WeightsNotFoundError
from trinetra.features.store import EmbeddingStore
from trinetra.parsers.models import Dataset, DatasetFormat, DatasetImage


def _write_solid_png(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (24, 16)) -> None:
    Image.new("RGB", size, color=color).save(path)


def _image_record(
    tmp_path: Path,
    image_id,
    file_name: str,
    color: tuple[int, int, int] | None,
) -> DatasetImage:
    file_path = None
    if color is not None:
        file_path = tmp_path / file_name
        _write_solid_png(file_path, color)
    return DatasetImage(
        image_id=image_id,
        file_name=file_name,
        width=24,
        height=16,
        file_path=file_path,
    )


def test_dummy_extractor_satisfies_protocol():
    extractor = DummyFeatureExtractor()
    assert isinstance(extractor, FeatureExtractor)


def test_backend_weight_errors_are_not_converted_to_image_failures():
    class MissingWeightsBackend:
        def extract(self, dataset):
            raise WeightsNotFoundError("required local weights are missing")

    dataset = Dataset(name="x", format=DatasetFormat.COCO, images=[])
    with pytest.raises(WeightsNotFoundError, match="required local weights"):
        MissingWeightsBackend().extract(dataset)


def test_dummy_extractor_is_deterministic_and_skips_unusable_images(tmp_path: Path):
    red_a = _image_record(tmp_path, 1, "red_a.png", (220, 10, 10))
    red_b = _image_record(tmp_path, "red_b", "red_b.png", (220, 10, 10))
    blue = _image_record(tmp_path, 2, "blue.png", (10, 10, 220))
    missing_path = DatasetImage(
        image_id=3,
        file_name="gone.png",
        width=24,
        height=16,
        file_path=tmp_path / "gone.png",
    )
    no_path = DatasetImage(
        image_id=4,
        file_name="unlinked.png",
        width=24,
        height=16,
        file_path=None,
    )
    corrupt = DatasetImage(
        image_id=5,
        file_name="corrupt.png",
        width=24,
        height=16,
        file_path=tmp_path / "corrupt.png",
    )
    (tmp_path / "corrupt.png").write_bytes(b"not-png")

    dataset = Dataset(
        name="slice1",
        format=DatasetFormat.COCO,
        images=[red_a, red_b, blue, missing_path, no_path, corrupt],
        categories={0: "n/a"},
    )

    extractor = DummyFeatureExtractor(embedding_dim=8)
    batch = extractor.extract(dataset)

    assert batch.backend == "dummy"
    assert batch.model_id == "dummy-meanpool-v1"
    assert batch.embedding_dim == 8
    assert batch.embedding_count == 3
    assert batch.embeddings[0].image_id == 1
    assert batch.embeddings[1].image_id == "red_b"
    assert batch.embeddings[2].image_id == 2
    np.testing.assert_allclose(batch.embeddings[0].vector, batch.embeddings[1].vector, rtol=1e-6, atol=1e-6)
    assert not np.allclose(batch.embeddings[0].vector, batch.embeddings[2].vector, rtol=1e-5, atol=1e-5)
    assert np.isfinite(batch.as_matrix()).all()
    assert len(batch.failures) == 3
    assert {failure.image_id for failure in batch.failures} == {3, 4, 5}
    assert {failure.failure_code for failure in batch.failures} == {
        "image_not_found", "image_path_missing", "unrecognized_image"
    }

    repeated = extractor.extract(dataset)
    np.testing.assert_array_equal(repeated.as_matrix(), batch.as_matrix())


def test_dummy_extractor_empty_dataset_does_not_invent_vectors():
    dataset = Dataset(name="empty", format=DatasetFormat.YOLO, images=[])
    batch = DummyFeatureExtractor(embedding_dim=4).extract(dataset)
    assert batch.embedding_count == 0
    assert batch.as_matrix().shape == (0, 4)
    assert batch.warnings == []
    assert batch.failures == []


def test_dummy_extractor_store_roundtrip(tmp_path: Path):
    images = [
        _image_record(tmp_path, 10, "a.png", (0, 0, 0)),
        _image_record(tmp_path, 11, "b.png", (255, 255, 255)),
    ]
    dataset = Dataset(name="persist", format=DatasetFormat.YOLO, images=images)
    batch = DummyFeatureExtractor(embedding_dim=6).extract(dataset)
    loaded = EmbeddingStore().load(EmbeddingStore().save(batch, tmp_path / "out"))
    assert loaded.embedding_count == 2
    np.testing.assert_array_equal(loaded.as_matrix(), batch.as_matrix())
    assert loaded.embeddings[0].image_id == 10
    assert loaded.embeddings[1].image_id == 11
