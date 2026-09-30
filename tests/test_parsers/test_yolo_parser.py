"""Tests for YOLO TXT dataset parser."""

import struct
from pathlib import Path

import pytest

from trinetra.parsers.exceptions import (
    BoundingBoxOutOfBoundsError,
    DatasetNotFoundError,
    InvalidAnnotationError,
    InvalidBoundingBoxError,
    MalformedDatasetError,
)
from trinetra.parsers.models import DatasetFormat
from trinetra.parsers.yolo import YoloParser


def create_dummy_png(path: Path, width: int = 640, height: int = 480):
    """Helper to create minimal valid PNG file for dimension extraction."""
    data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\r"
        b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )
    path.write_bytes(data)


@pytest.fixture
def yolo_structure(tmp_path: Path):
    data_dir = tmp_path / "dataset"
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Classes
    classes_file = data_dir / "classes.txt"
    classes_file.write_text("drone\nairplane\nhelicopter\n", encoding="utf-8")

    # Images (PNG 640x480)
    create_dummy_png(images_dir / "img1.png", width=640, height=480)
    create_dummy_png(images_dir / "img2.png", width=800, height=600)
    create_dummy_png(images_dir / "img3.png", width=640, height=480)

    # Label for img1: 2 annotations
    # drone: cx=0.5, cy=0.5, w=0.5, h=0.5 (centered box)
    # airplane: cx=0.2, cy=0.3, w=0.1, h=0.2
    (labels_dir / "img1.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n1 0.2 0.3 0.1 0.2\n",
        encoding="utf-8",
    )

    # Label for img2: empty file (0 annotations)
    (labels_dir / "img2.txt").write_text("", encoding="utf-8")

    # img3 has no label file (background image with 0 annotations)

    return data_dir


def test_yolo_parser_success(yolo_structure: Path):
    parser = YoloParser()
    dataset = parser.parse(yolo_structure)

    assert dataset.format == DatasetFormat.YOLO
    assert dataset.image_count == 3
    assert dataset.annotation_count == 2
    assert dataset.categories == {0: "drone", 1: "airplane", 2: "helicopter"}

    # Image 1
    img1 = dataset.get_image("img1")
    assert img1 is not None
    assert img1.width == 640
    assert img1.height == 480
    assert img1.annotation_count == 2

    # Annotation 1: class 0 (drone), centered 0.5, 0.5, w=0.5, h=0.5 on 640x480
    ann1 = img1.annotations[0]
    assert ann1.category_id == 0
    assert ann1.category_name == "drone"
    assert ann1.bbox.xmin == 160.0
    assert ann1.bbox.ymin == 120.0
    assert ann1.bbox.width == 320.0
    assert ann1.bbox.height == 240.0
    assert ann1.bbox.normalized == (0.5, 0.5, 0.5, 0.5)

    # Annotation 2: class 1 (airplane)
    ann2 = img1.annotations[1]
    assert ann2.category_id == 1
    assert ann2.category_name == "airplane"
    assert ann2.bbox.normalized == (0.2, 0.3, 0.1, 0.2)

    # Image 2 (empty label file)
    img2 = dataset.get_image("img2")
    assert img2 is not None
    assert img2.annotation_count == 0

    # Image 3 (no label file)
    img3 = dataset.get_image("img3")
    assert img3 is not None
    assert img3.annotation_count == 0


def test_yolo_asymmetric_coordinates_protection(tmp_path: Path):
    """Specifically tests asymmetric dimensions (1920x1080) to protect against width/height swapping."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "hd_frame.png", width=1920, height=1080)
    # cx=0.2, cy=0.7, w=0.1, h=0.3
    # pixel_w = 0.1 * 1920 = 192.0
    # pixel_h = 0.3 * 1080 = 324.0
    # pixel_xmin = (0.2 - 0.1/2) * 1920 = 0.15 * 1920 = 288.0
    # pixel_ymin = (0.7 - 0.3/2) * 1080 = 0.55 * 1080 = 594.0
    (labels_dir / "hd_frame.txt").write_text("0 0.2 0.7 0.1 0.3\n", encoding="utf-8")

    parser = YoloParser()
    dataset = parser.parse(tmp_path, classes=["vehicle"])

    img = dataset.get_image("hd_frame")
    assert img is not None
    assert img.width == 1920
    assert img.height == 1080

    ann = img.annotations[0]
    assert ann.bbox.xmin == pytest.approx(288.0)
    assert ann.bbox.ymin == pytest.approx(594.0)
    assert ann.bbox.width == pytest.approx(192.0)
    assert ann.bbox.height == pytest.approx(324.0)
    assert ann.bbox.xmax == pytest.approx(480.0)
    assert ann.bbox.ymax == pytest.approx(918.0)
    assert ann.bbox.normalized == (0.2, 0.7, 0.1, 0.3)


def test_yolo_orphan_label_detection(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    # Image 1 exists
    create_dummy_png(images_dir / "img1.png", 640, 480)
    (labels_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Orphan label file exists (no matching image img2.png)
    (labels_dir / "orphan_label.txt").write_text("0 0.1 0.1 0.1 0.1\n", encoding="utf-8")

    # Standard classes.txt should NOT be treated as orphan label!
    (labels_dir / "classes.txt").write_text("target\n", encoding="utf-8")

    parser = YoloParser()

    # Non-strict mode: warns in dataset.warnings
    dataset = parser.parse(tmp_path, strict_orphans=False)
    assert dataset.image_count == 1
    assert any("Orphan label file without corresponding image: orphan_label.txt" in w for w in dataset.warnings)

    # Strict mode: raises InvalidAnnotationError
    with pytest.raises(InvalidAnnotationError, match="Orphan label file"):
        parser.parse(tmp_path, strict_orphans=True)


def test_yolo_missing_classes_warning(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "test.png", 640, 480)
    (labels_dir / "test.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    # No classes.txt created, no classes passed

    parser = YoloParser()
    dataset = parser.parse(tmp_path)

    assert dataset.image_count == 1
    assert dataset.annotation_count == 1
    assert dataset.images[0].annotations[0].category_name is None
    assert any("No class mapping found" in w for w in dataset.warnings)


def test_yolo_parser_programmatic_classes(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "target.png", 500, 500)
    (labels_dir / "target.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    parser = YoloParser()
    dataset = parser.parse(
        tmp_path,
        classes=["first_class", "second_class"],
    )

    img = dataset.get_image("target")
    assert img is not None
    assert img.annotations[0].category_id == 1
    assert img.annotations[0].category_name == "second_class"
    assert len(dataset.warnings) == 0


def test_yolo_boundary_validation_non_strict_preserves_coordinates(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "overflow.png", 640, 480)
    # cx=0.95, w=0.2 -> xmax_norm = 0.95 + 0.1 = 1.05 (out of bounds)
    (labels_dir / "overflow.txt").write_text("0 0.95 0.5 0.2 0.4\n", encoding="utf-8")

    parser = YoloParser()
    dataset = parser.parse(tmp_path, classes=["object"], strict_bounds=False)

    img = dataset.get_image("overflow")
    assert img is not None
    ann = img.annotations[0]

    # Preserves coordinates without clipping!
    assert ann.bbox.normalized == (0.95, 0.5, 0.2, 0.4)
    assert ann.bbox.xmin == pytest.approx((0.95 - 0.1) * 640)
    assert len(ann.warnings) > 0
    assert "extends outside normalized" in ann.warnings[0]
    assert len(img.warnings) > 0


def test_yolo_boundary_validation_strict_raises(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "overflow.png", 640, 480)
    (labels_dir / "overflow.txt").write_text("0 0.95 0.5 0.2 0.4\n", encoding="utf-8")

    parser = YoloParser()
    with pytest.raises(BoundingBoxOutOfBoundsError, match="extends outside normalized"):
        parser.parse(tmp_path, classes=["object"], strict_bounds=True)


def test_yolo_missing_directory():
    parser = YoloParser()
    with pytest.raises(DatasetNotFoundError):
        parser.parse("nonexistent_directory")


def test_yolo_malformed_line_token_count(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "bad.png", 640, 480)
    # Only 3 tokens instead of 5
    (labels_dir / "bad.txt").write_text("0 0.5 0.5\n", encoding="utf-8")

    parser = YoloParser()
    with pytest.raises(MalformedDatasetError, match="expected 5 tokens"):
        parser.parse(tmp_path, classes=["c"])


def test_yolo_malformed_line_non_numeric(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "bad.png", 640, 480)
    (labels_dir / "bad.txt").write_text("0 0.5 abc 0.2 0.2\n", encoding="utf-8")

    parser = YoloParser()
    with pytest.raises(MalformedDatasetError, match="Non-numeric coordinates"):
        parser.parse(tmp_path, classes=["c"])


def test_yolo_invalid_class_id_negative(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "bad.png", 640, 480)
    (labels_dir / "bad.txt").write_text("-1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    parser = YoloParser()
    with pytest.raises(InvalidAnnotationError, match="Invalid negative class ID"):
        parser.parse(tmp_path, classes=["c"])


def test_yolo_invalid_class_id_unknown(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "bad.png", 640, 480)
    # Class ID 5 but only 1 class (index 0) defined
    (labels_dir / "bad.txt").write_text("5 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    parser = YoloParser()
    with pytest.raises(InvalidAnnotationError, match="not found in known categories"):
        parser.parse(tmp_path, classes=["single_class"])


def test_yolo_non_positive_box_dimensions(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()

    create_dummy_png(images_dir / "bad.png", 640, 480)
    (labels_dir / "bad.txt").write_text("0 0.5 0.5 -0.2 0.2\n", encoding="utf-8")

    parser = YoloParser()
    with pytest.raises(InvalidBoundingBoxError, match="Non-positive bounding box"):
        parser.parse(tmp_path, classes=["c"])
