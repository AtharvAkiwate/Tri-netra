"""Common Dataset Representation for TRI-NETRA.

Standardized internal representation of datasets, images, and annotations,
allowing downstream security detectors to operate independently of source
dataset formats (COCO JSON, YOLO TXT, etc.).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from trinetra.parsers.exceptions import InvalidBoundingBoxError


class DatasetFormat(str, Enum):
    """Supported dataset formats."""
    COCO = "coco"
    YOLO = "yolo"


@dataclass
class BoundingBox:
    """Canonical bounding box representation in absolute pixel coordinates (xywh).

    Attributes:
        xmin: Top-left x-coordinate in absolute pixels.
        ymin: Top-left y-coordinate in absolute pixels.
        width: Bounding box width in absolute pixels. Must be non-negative.
        height: Bounding box height in absolute pixels. Must be non-negative.
        normalized: Optional original YOLO normalized coordinates (center_x, center_y, width, height)
                    where values are relative to image dimensions in [0.0, 1.0].
    """
    xmin: float
    ymin: float
    width: float
    height: float
    normalized: Optional[tuple[float, float, float, float]] = None

    def __post_init__(self) -> None:
        """Validate bounding box dimensions and numeric values."""
        for val, name in [
            (self.xmin, "xmin"),
            (self.ymin, "ymin"),
            (self.width, "width"),
            (self.height, "height"),
        ]:
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise InvalidBoundingBoxError(
                    f"Bounding box {name} must be a finite number, got {val!r}"
                )

        if self.width < 0:
            raise InvalidBoundingBoxError(f"Bounding box width cannot be negative: {self.width}")
        if self.height < 0:
            raise InvalidBoundingBoxError(f"Bounding box height cannot be negative: {self.height}")

        if self.normalized is not None:
            if not isinstance(self.normalized, tuple) or len(self.normalized) != 4:
                raise InvalidBoundingBoxError(
                    f"Normalized bounding box must be a 4-tuple, got {self.normalized!r}"
                )
            for val in self.normalized:
                if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                    raise InvalidBoundingBoxError(
                        f"Normalized bounding box values must be finite numbers, got {val!r}"
                    )

    @property
    def xmax(self) -> float:
        """Maximum x-coordinate (bottom-right x) in pixels."""
        return self.xmin + self.width

    @property
    def ymax(self) -> float:
        """Maximum y-coordinate (bottom-right y) in pixels."""
        return self.ymin + self.height

    @property
    def center_x(self) -> float:
        """Center x-coordinate in pixels."""
        return self.xmin + self.width / 2.0

    @property
    def center_y(self) -> float:
        """Center y-coordinate in pixels."""
        return self.ymin + self.height / 2.0

    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return canonical pixel representation as (xmin, ymin, width, height)."""
        return (self.xmin, self.ymin, self.width, self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Return pixel representation as (xmin, ymin, xmax, ymax)."""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def as_clamped_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Return non-destructive pixel coordinates clamped to valid image raster boundaries [0, img_w] x [0, img_h].

        Leaves canonical coordinates untouched. Useful for safe tensor cropping.

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            Tuple of (clamped_xmin, clamped_ymin, clamped_xmax, clamped_ymax).
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={img_w}, height={img_h}")
        c_xmin = max(0.0, min(float(img_w), self.xmin))
        c_ymin = max(0.0, min(float(img_h), self.ymin))
        c_xmax = max(0.0, min(float(img_w), self.xmax))
        c_ymax = max(0.0, min(float(img_h), self.ymax))
        return (c_xmin, c_ymin, c_xmax, c_ymax)

    def as_clamped_xywh(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Return non-destructive pixel coordinates clamped to image raster boundaries as (xmin, ymin, width, height).

        Leaves canonical coordinates untouched. Useful for safe tensor cropping.

        Args:
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            Tuple of (clamped_xmin, clamped_ymin, clamped_width, clamped_height).
        """
        c_xmin, c_ymin, c_xmax, c_ymax = self.as_clamped_xyxy(img_w, img_h)
        return (c_xmin, c_ymin, max(0.0, c_xmax - c_xmin), max(0.0, c_ymax - c_ymin))

    def as_yolo_normalized(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert pixel bounding box to YOLO normalized format (center_x, center_y, width, height).

        Args:
            img_w: Image width in pixels. Must be positive.
            img_h: Image height in pixels. Must be positive.

        Returns:
            Tuple of (center_x, center_y, width, height) normalized to [0.0, 1.0].
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={img_w}, height={img_h}")
        return (
            self.center_x / img_w,
            self.center_y / img_h,
            self.width / img_w,
            self.height / img_h,
        )

    @classmethod
    def from_xywh(
        cls,
        xmin: float,
        ymin: float,
        width: float,
        height: float,
        normalized: Optional[tuple[float, float, float, float]] = None,
    ) -> BoundingBox:
        """Create BoundingBox from canonical pixel (xmin, ymin, width, height)."""
        return cls(
            xmin=float(xmin),
            ymin=float(ymin),
            width=float(width),
            height=float(height),
            normalized=normalized,
        )

    @classmethod
    def from_yolo(
        cls,
        cx: float,
        cy: float,
        w: float,
        h: float,
        img_w: int,
        img_h: int,
    ) -> BoundingBox:
        """Create BoundingBox from YOLO normalized coordinates (center_x, center_y, width, height).

        Args:
            cx: Normalized center x-coordinate in [0.0, 1.0].
            cy: Normalized center y-coordinate in [0.0, 1.0].
            w: Normalized width in [0.0, 1.0].
            h: Normalized height in [0.0, 1.0].
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            BoundingBox instance with absolute pixel coordinates, preserving normalized tuple.
        """
        if img_w <= 0 or img_h <= 0:
            raise ValueError(f"Image dimensions must be positive, got width={img_w}, height={img_h}")

        pixel_w = float(w * img_w)
        pixel_h = float(h * img_h)
        pixel_xmin = float((cx - w / 2.0) * img_w)
        pixel_ymin = float((cy - h / 2.0) * img_h)

        return cls(
            xmin=pixel_xmin,
            ymin=pixel_ymin,
            width=pixel_w,
            height=pixel_h,
            normalized=(float(cx), float(cy), float(w), float(h)),
        )


@dataclass
class Annotation:
    """Represents a single object annotation within an image.

    Attributes:
        annotation_id: Unique identifier for the annotation (int or str).
        category_id: Integer category/class identifier.
        category_name: Human-readable name of the category, when available.
        bbox: Bounding box in canonical representation.
        area: Annotation area in square pixels, when available.
        iscrowd: COCO crowd flag (0 = single object, 1 = crowd).
        warnings: List of non-fatal validation diagnostics (e.g. boundary spillover).
    """
    annotation_id: Union[int, str]
    category_id: int
    bbox: BoundingBox
    category_name: Optional[str] = None
    area: Optional[float] = None
    iscrowd: Optional[int] = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DatasetImage:
    """Represents an image in the dataset along with its annotations.

    Attributes:
        image_id: Unique identifier for the image (int or str).
        file_name: Image filename (e.g. '000001.jpg').
        width: Image width in pixels.
        height: Image height in pixels.
        annotations: List of annotations associated with this image.
        warnings: List of non-fatal validation diagnostics for this image.
        file_path: Optional resolved local Path to the image file on disk.
    """
    image_id: Union[int, str]
    file_name: str
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    file_path: Optional[Path] = None

    @property
    def annotation_count(self) -> int:
        """Return the number of annotations for this image."""
        return len(self.annotations)


@dataclass
class Dataset:
    """Unified internal representation of an object detection dataset.

    Attributes:
        name: Name of the dataset.
        format: DatasetFormat enum indicating original format (COCO, YOLO).
        images: List of DatasetImage objects.
        categories: Mapping of category_id (int) to category name (str).
        warnings: Dataset-level validation warnings.
        root_path: Optional Path to the dataset root directory on disk.
    """
    name: str
    format: DatasetFormat
    images: list[DatasetImage] = field(default_factory=list)
    categories: dict[int, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    root_path: Optional[Path] = None
    _image_map: Optional[dict[Union[int, str], DatasetImage]] = field(default=None, init=False, repr=False)

    def _ensure_image_map(self) -> dict[Union[int, str], DatasetImage]:
        """Build or refresh fast O(1) image lookup map."""
        if self._image_map is None or len(self._image_map) != len(self.images):
            self._image_map = {img.image_id: img for img in self.images}
        return self._image_map

    @property
    def image_count(self) -> int:
        """Total number of images in the dataset."""
        return len(self.images)

    @property
    def annotation_count(self) -> int:
        """Total number of annotations across all images in the dataset."""
        return sum(img.annotation_count for img in self.images)

    def get_image(self, image_id: Union[int, str]) -> Optional[DatasetImage]:
        """Look up an image by its unique ID in O(1) time.

        Supports direct key lookup first. If not found, falls back safely to
        int/string equivalence when unambiguous. Returns None if not found.
        """
        img_map = self._ensure_image_map()

        # 1. Exact match (O(1))
        if image_id in img_map:
            return img_map[image_id]

        # 2. String representation fallback for integer image_id
        if isinstance(image_id, int):
            str_key = str(image_id)
            if str_key in img_map:
                return img_map[str_key]

        # 3. Integer representation fallback for numeric string image_id
        elif isinstance(image_id, str) and image_id.isdigit():
            try:
                int_key = int(image_id)
                if int_key in img_map:
                    return img_map[int_key]
            except ValueError:
                pass

        return None

    def get_category_name(self, category_id: int) -> Optional[str]:
        """Look up category name by its ID. Returns None if not found."""
        return self.categories.get(category_id)
