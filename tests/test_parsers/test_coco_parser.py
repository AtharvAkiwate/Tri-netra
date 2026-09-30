"""Tests for COCO JSON dataset parser."""

import json
from pathlib import Path

import pytest

from trinetra.parsers.coco import CocoParser
from trinetra.parsers.exceptions import (
    BoundingBoxOutOfBoundsError,
    DatasetNotFoundError,
    InvalidAnnotationError,
    InvalidBoundingBoxError,
    MalformedDatasetError,
)
from trinetra.parsers.models import DatasetFormat


@pytest.fixture
def valid_coco_dict():
    return {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 800, "height": 600},
            {"id": 3, "file_name": "img3.jpg", "width": 1024, "height": 768},
        ],
        "categories": [
            {"id": 1, "name": "vehicle"},
            {"id": 2, "name": "pedestrian"},
        ],
        "annotations": [
            {
                "id": 101,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 50.0, 60.0],
                "area": 3000.0,
                "iscrowd": 0,
            },
            {
                "id": 102,
                "image_id": 1,
                "category_id": 2,
                "bbox": [100.0, 150.0, 30.0, 80.0],
                "area": 2400.0,
                "iscrowd": 0,
            },
            {
                "id": 103,
                "image_id": 2,
                "category_id": 1,
                "bbox": [200.0, 250.0, 40.0, 70.0],
                "area": 2800.0,
                "iscrowd": 1,
            },
        ],
    }


def test_coco_parser_success(tmp_path: Path, valid_coco_dict):
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(valid_coco_dict), encoding="utf-8")

    parser = CocoParser()
    dataset = parser.parse(ann_file)

    assert dataset.format == DatasetFormat.COCO
    assert dataset.image_count == 3
    assert dataset.annotation_count == 3
    assert dataset.categories == {1: "vehicle", 2: "pedestrian"}

    # Image 1 has 2 annotations
    img1 = dataset.get_image(1)
    assert img1 is not None
    assert img1.annotation_count == 2
    assert img1.file_name == "img1.jpg"
    assert img1.width == 640
    assert img1.height == 480

    ann1 = img1.annotations[0]
    assert ann1.annotation_id == 101
    assert ann1.category_id == 1
    assert ann1.category_name == "vehicle"
    assert ann1.bbox.as_xywh() == (10.0, 20.0, 50.0, 60.0)
    assert ann1.area == 3000.0
    assert ann1.iscrowd == 0
    assert len(ann1.warnings) == 0

    # Image 2 has 1 annotation
    img2 = dataset.get_image(2)
    assert img2 is not None
    assert img2.annotation_count == 1
    assert img2.annotations[0].iscrowd == 1

    # Image 3 has 0 annotations
    img3 = dataset.get_image(3)
    assert img3 is not None
    assert img3.annotation_count == 0


def test_coco_local_image_dir_fallback(tmp_path: Path, valid_coco_dict):
    """When image_dir is omitted, parser should resolve parent/images or parent."""
    # Scenario A: parent / "images" directory exists
    dataset_dir = tmp_path / "coco_set"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    ann_file = dataset_dir / "annotations.json"
    ann_file.write_text(json.dumps(valid_coco_dict), encoding="utf-8")

    parser = CocoParser()
    dataset = parser.parse(ann_file)  # image_dir omitted

    img1 = dataset.get_image(1)
    assert img1 is not None
    assert img1.file_path == images_dir / "img1.jpg"

    # Scenario B: explicit image_dir provided overrides fallback
    custom_dir = tmp_path / "custom_images"
    custom_dir.mkdir()
    dataset_custom = parser.parse(ann_file, image_dir=custom_dir)
    assert dataset_custom.get_image(1).file_path == custom_dir / "img1.jpg"


def test_coco_boundary_validation_non_strict_preserves_coordinates(tmp_path: Path):
    # Image is 100x100, but bbox has width 150 extending outside boundary
    coco_data = {
        "images": [{"id": 1, "file_name": "overflow.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "box"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 150.0, 50.0],
            }
        ],
    }
    ann_file = tmp_path / "overflow.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    parser = CocoParser()
    dataset = parser.parse(ann_file, strict_bounds=False)

    img = dataset.get_image(1)
    assert img is not None
    ann = img.annotations[0]

    # Coordinates must be preserved without clipping!
    assert ann.bbox.as_xywh() == (10.0, 10.0, 150.0, 50.0)
    assert len(ann.warnings) > 0
    assert "extends outside image bounds" in ann.warnings[0]
    assert len(img.warnings) > 0


def test_coco_boundary_validation_strict_raises(tmp_path: Path):
    coco_data = {
        "images": [{"id": 1, "file_name": "overflow.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "box"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 10.0, 150.0, 50.0],
            }
        ],
    }
    ann_file = tmp_path / "overflow.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(BoundingBoxOutOfBoundsError, match="extends outside image bounds"):
        parser.parse(ann_file, strict_bounds=True)


def test_coco_missing_file():
    parser = CocoParser()
    with pytest.raises(DatasetNotFoundError):
        parser.parse("nonexistent_annotations.json")


def test_coco_malformed_json(tmp_path: Path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ unclosed json: ", encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(MalformedDatasetError, match="Malformed JSON"):
        parser.parse(bad_json)


def test_coco_missing_required_keys(tmp_path: Path):
    missing_keys_json = tmp_path / "missing.json"
    missing_keys_json.write_text(json.dumps({"images": []}), encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(MalformedDatasetError, match="missing required top-level keys"):
        parser.parse(missing_keys_json)


def test_coco_duplicate_image_id(tmp_path: Path):
    coco_data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 100},
            {"id": 1, "file_name": "b.jpg", "width": 100, "height": 100},
        ],
        "categories": [],
        "annotations": [],
    }
    ann_file = tmp_path / "duplicate_img.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(MalformedDatasetError, match="Duplicate image id"):
        parser.parse(ann_file)


def test_coco_invalid_image_reference(tmp_path: Path):
    coco_data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 999, "category_id": 1, "bbox": [0, 0, 10, 10]}
        ],
    }
    ann_file = tmp_path / "invalid_img_ref.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(InvalidAnnotationError, match="references nonexistent image_id"):
        parser.parse(ann_file)


def test_coco_invalid_category_reference(tmp_path: Path):
    coco_data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 999, "bbox": [0, 0, 10, 10]}
        ],
    }
    ann_file = tmp_path / "invalid_cat_ref.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(InvalidAnnotationError, match="references nonexistent category_id"):
        parser.parse(ann_file)


def test_coco_invalid_bbox_values(tmp_path: Path):
    # Negative width
    coco_data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, -10, 10]}
        ],
    }
    ann_file = tmp_path / "neg_bbox.json"
    ann_file.write_text(json.dumps(coco_data), encoding="utf-8")

    parser = CocoParser()
    with pytest.raises(InvalidBoundingBoxError, match="width cannot be negative"):
        parser.parse(ann_file)
