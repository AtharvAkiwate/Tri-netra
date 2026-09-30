"""Tests for common dataset representation models."""

import pytest

from trinetra.parsers.exceptions import InvalidBoundingBoxError
from trinetra.parsers.models import (
    Annotation,
    BoundingBox,
    Dataset,
    DatasetFormat,
    DatasetImage,
)


def test_bounding_box_creation_and_properties():
    bbox = BoundingBox(xmin=10.0, ymin=20.0, width=50.0, height=60.0)

    assert bbox.xmin == 10.0
    assert bbox.ymin == 20.0
    assert bbox.width == 50.0
    assert bbox.height == 60.0
    assert bbox.xmax == 60.0
    assert bbox.ymax == 80.0
    assert bbox.center_x == 35.0
    assert bbox.center_y == 50.0
    assert bbox.as_xywh() == (10.0, 20.0, 50.0, 60.0)
    assert bbox.as_xyxy() == (10.0, 20.0, 60.0, 80.0)


def test_bounding_box_clamped_helpers_preserve_original():
    # Box extending outside image bounds: xmin=-20 (left spill), width=100, xmax=80
    # ymin=400, height=150 -> ymax=550 (bottom spill on 640x480)
    bbox = BoundingBox(
        xmin=-20.0,
        ymin=400.0,
        width=100.0,
        height=150.0,
        normalized=(-0.03125, 0.98958, 0.15625, 0.3125),
    )

    clamped_xyxy = bbox.as_clamped_xyxy(img_w=640, img_h=480)
    assert clamped_xyxy == (0.0, 400.0, 80.0, 480.0)

    clamped_xywh = bbox.as_clamped_xywh(img_w=640, img_h=480)
    assert clamped_xywh == (0.0, 400.0, 80.0, 80.0)

    # CRITICAL: Canonical attributes MUST remain completely unchanged!
    assert bbox.xmin == -20.0
    assert bbox.ymin == 400.0
    assert bbox.width == 100.0
    assert bbox.height == 150.0
    assert bbox.normalized == (-0.03125, 0.98958, 0.15625, 0.3125)

    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        bbox.as_clamped_xyxy(img_w=0, img_h=480)


def test_bounding_box_yolo_conversion_and_preservation():
    # Image 640x480, box centered at (0.5, 0.5) with width=0.5 (320), height=0.5 (240)
    bbox = BoundingBox.from_yolo(
        cx=0.5,
        cy=0.5,
        w=0.5,
        h=0.5,
        img_w=640,
        img_h=480,
    )

    assert bbox.xmin == 160.0
    assert bbox.ymin == 120.0
    assert bbox.width == 320.0
    assert bbox.height == 240.0
    assert bbox.normalized == (0.5, 0.5, 0.5, 0.5)
    assert bbox.as_yolo_normalized(640, 480) == (0.5, 0.5, 0.5, 0.5)


def test_bounding_box_invalid_dimensions():
    with pytest.raises(InvalidBoundingBoxError, match="width cannot be negative"):
        BoundingBox(xmin=0.0, ymin=0.0, width=-10.0, height=20.0)

    with pytest.raises(InvalidBoundingBoxError, match="height cannot be negative"):
        BoundingBox(xmin=0.0, ymin=0.0, width=10.0, height=-20.0)

    with pytest.raises(InvalidBoundingBoxError, match="must be a finite number"):
        BoundingBox(xmin=float("nan"), ymin=0.0, width=10.0, height=20.0)

    with pytest.raises(InvalidBoundingBoxError, match="must be a finite number"):
        BoundingBox(xmin=0.0, ymin=float("inf"), width=10.0, height=20.0)


def test_dataset_and_image_aggregations():
    bbox = BoundingBox(xmin=5.0, ymin=5.0, width=10.0, height=10.0)
    ann1 = Annotation(annotation_id=1, category_id=0, category_name="person", bbox=bbox)
    ann2 = Annotation(annotation_id=2, category_id=1, category_name="car", bbox=bbox)

    img1 = DatasetImage(
        image_id=101,
        file_name="img1.jpg",
        width=640,
        height=480,
        annotations=[ann1, ann2],
    )
    img2 = DatasetImage(
        image_id="frame_002",
        file_name="img2.jpg",
        width=800,
        height=600,
        annotations=[],
    )

    dataset = Dataset(
        name="test_dataset",
        format=DatasetFormat.COCO,
        images=[img1, img2],
        categories={0: "person", 1: "car"},
        warnings=["Sample warning"],
    )

    assert dataset.image_count == 2
    assert dataset.annotation_count == 2
    assert img1.annotation_count == 2
    assert img2.annotation_count == 0
    assert dataset.warnings == ["Sample warning"]

    # Exact key lookups (O(1))
    assert dataset.get_image(101) == img1
    assert dataset.get_image("frame_002") == img2

    # String-to-int / int-to-string fallback lookups
    assert dataset.get_image("101") == img1
    assert dataset.get_image(999) is None
    assert dataset.get_category_name(0) == "person"
    assert dataset.get_category_name(99) is None
