"""Content-change invalidation checks for feature caches."""

from pathlib import Path

from PIL import Image

from trinetra.features.cache import fingerprint_dataset
from trinetra.parsers.models import Dataset, DatasetFormat, DatasetImage


def test_dataset_fingerprint_changes_when_source_image_bytes_change(tmp_path: Path):
    path = tmp_path / "same-name.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(path)
    dataset = Dataset("d", DatasetFormat.COCO, images=[
        DatasetImage(1, path.name, 8, 8, file_path=path)
    ])
    before = fingerprint_dataset(dataset)
    Image.new("RGB", (8, 8), (255, 0, 0)).save(path)
    after = fingerprint_dataset(dataset)
    assert before != after
