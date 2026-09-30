"""Tests for unified load_dataset API and agnostic Layer 1 consumption."""

import json
import struct
from pathlib import Path

import pytest

from trinetra.parsers import load_dataset
from trinetra.parsers.exceptions import DatasetNotFoundError, TrinetraParserError
from trinetra.parsers.models import Dataset, DatasetFormat


def create_dummy_png(path: Path, width: int = 640, height: int = 480):
    data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\r"
        b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    path.write_bytes(data)


def test_load_dataset_coco_autodetect(tmp_path: Path):
    coco_data = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 480}],
        "categories": [{"id": 1, "name": "target"}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40]}
        ],
    }
    json_path = tmp_path / "coco_annotations.json"
    json_path.write_text(json.dumps(coco_data), encoding="utf-8")

    dataset = load_dataset(json_path)

    assert isinstance(dataset, Dataset)
    assert dataset.format == DatasetFormat.COCO
    assert dataset.image_count == 1
    assert dataset.annotation_count == 1
    assert dataset.images[0].annotations[0].bbox.as_xywh() == (10.0, 20.0, 30.0, 40.0)


def test_load_dataset_yolo_autodetect(tmp_path: Path):
    yolo_dir = tmp_path / "yolo_data"
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    create_dummy_png(images_dir / "frame01.png", 640, 480)
    (labels_dir / "frame01.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (yolo_dir / "classes.txt").write_text("target\n", encoding="utf-8")

    dataset = load_dataset(yolo_dir)

    assert isinstance(dataset, Dataset)
    assert dataset.format == DatasetFormat.YOLO
    assert dataset.image_count == 1
    assert dataset.annotation_count == 1


def test_load_dataset_explicit_format(tmp_path: Path):
    coco_data = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    custom_named_file = tmp_path / "annotations.dat"
    custom_named_file.write_text(json.dumps(coco_data), encoding="utf-8")

    dataset = load_dataset(custom_named_file, format="coco")
    assert dataset.format == DatasetFormat.COCO
    assert dataset.image_count == 1


def test_load_dataset_directory_with_explicit_coco_format(tmp_path: Path):
    coco_dir = tmp_path / "coco_dir"
    coco_dir.mkdir()
    coco_data = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [],
    }
    (coco_dir / "dataset.json").write_text(json.dumps(coco_data), encoding="utf-8")

    # Directory with single JSON should resolve
    dataset = load_dataset(coco_dir, format="coco")
    assert dataset.format == DatasetFormat.COCO
    assert dataset.image_count == 1

    # Directory with 0 JSON files should raise DatasetNotFoundError
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(DatasetNotFoundError, match="No COCO JSON annotation files"):
        load_dataset(empty_dir, format="coco")

    # Directory with multiple ambiguous JSON files should fail clearly
    (coco_dir / "second_dataset.json").write_text(json.dumps(coco_data), encoding="utf-8")
    with pytest.raises(TrinetraParserError, match="Ambiguous COCO annotation files"):
        load_dataset(coco_dir, format="coco")


def test_load_dataset_errors(tmp_path: Path):
    with pytest.raises(DatasetNotFoundError):
        load_dataset("nonexistent_path")

    dummy_file = tmp_path / "unknown.bin"
    dummy_file.write_bytes(b"data")
    with pytest.raises(TrinetraParserError, match="Cannot automatically determine dataset format"):
        load_dataset(dummy_file)

    with pytest.raises(TrinetraParserError, match="Unsupported dataset format"):
        load_dataset(dummy_file, format="unsupported_format")


def test_agnostic_layer1_detector_consumer(tmp_path: Path):
    """Verify that a generic downstream detector can consume both COCO and YOLO datasets identically."""

    # 1. Prepare COCO dataset
    coco_json = tmp_path / "data.json"
    coco_json.write_text(
        json.dumps({
            "images": [{"id": 1, "file_name": "coco.jpg", "width": 640, "height": 480}],
            "categories": [{"id": 0, "name": "box"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [100, 100, 200, 200]}],
        }),
        encoding="utf-8",
    )

    # 2. Prepare YOLO dataset
    yolo_dir = tmp_path / "yolo"
    (yolo_dir / "images").mkdir(parents=True)
    (yolo_dir / "labels").mkdir(parents=True)
    create_dummy_png(yolo_dir / "images" / "yolo.png", 640, 480)
    # Box centered at 200, 200 with width 200, height 200 on 640x480:
    # cx = 200/640 = 0.3125, cy = 200/480 = 0.4166667, w = 200/640 = 0.3125, h = 200/480 = 0.4166667
    (yolo_dir / "labels" / "yolo.txt").write_text("0 0.3125 0.4166666666666667 0.3125 0.4166666666666667\n", encoding="utf-8")
    (yolo_dir / "classes.txt").write_text("box\n", encoding="utf-8")

    coco_dataset = load_dataset(coco_json)
    yolo_dataset = load_dataset(yolo_dir)

    def dummy_detector_engine(dataset: Dataset) -> list[dict]:
        """Conceptual detector engine operating purely on the common representation."""
        results = []
        for img in dataset.images:
            for ann in img.annotations:
                xmin, ymin, w, h = ann.bbox.as_xywh()
                results.append({
                    "file_name": img.file_name,
                    "category": ann.category_name,
                    "bbox_area": ann.bbox.width * ann.bbox.height,
                    "center": (ann.bbox.center_x, ann.bbox.center_y),
                })
        return results

    coco_results = dummy_detector_engine(coco_dataset)
    yolo_results = dummy_detector_engine(yolo_dataset)

    assert len(coco_results) == 1
    assert len(yolo_results) == 1

    assert coco_results[0]["category"] == "box"
    assert yolo_results[0]["category"] == "box"

    assert coco_results[0]["bbox_area"] == pytest.approx(40000.0)
    assert yolo_results[0]["bbox_area"] == pytest.approx(40000.0, rel=1e-3)
    assert coco_results[0]["center"] == pytest.approx((200.0, 200.0))
    assert yolo_results[0]["center"] == pytest.approx((200.0, 200.0), rel=1e-3)
