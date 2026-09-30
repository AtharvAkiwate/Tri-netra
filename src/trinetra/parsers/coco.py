"""COCO format dataset parser for TRI-NETRA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

from trinetra.parsers.exceptions import (
    BoundingBoxOutOfBoundsError,
    DatasetNotFoundError,
    InvalidAnnotationError,
    InvalidBoundingBoxError,
    MalformedDatasetError,
)
from trinetra.parsers.models import (
    Annotation,
    BoundingBox,
    Dataset,
    DatasetFormat,
    DatasetImage,
)


class CocoParser:
    """Parser for COCO JSON format datasets."""

    def parse(
        self,
        annotation_file: Union[str, Path],
        image_dir: Optional[Union[str, Path]] = None,
        dataset_name: Optional[str] = None,
        strict_bounds: bool = False,
    ) -> Dataset:
        """Parse a COCO JSON dataset file into the common representation.

        Args:
            annotation_file: Path to the COCO JSON annotations file.
            image_dir: Optional path to the directory containing corresponding image files.
                       If omitted, checks local candidate directories (parent/images, parent).
            dataset_name: Optional human-readable dataset name. Defaults to stem of annotation_file.
            strict_bounds: If True, raises BoundingBoxOutOfBoundsError when a bounding box
                           extends outside image boundaries. If False (default), preserves original
                           coordinates without clipping and records a diagnostic warning.

        Returns:
            Dataset: Standardized Common Dataset Representation.

        Raises:
            DatasetNotFoundError: If annotation_file does not exist.
            MalformedDatasetError: If JSON is invalid or missing required top-level keys.
            InvalidAnnotationError: If an annotation references an invalid image or category ID.
            InvalidBoundingBoxError: If bounding box contains negative dimensions or non-numeric values.
            BoundingBoxOutOfBoundsError: If strict_bounds=True and a bounding box exceeds image dimensions.
        """
        ann_path = Path(annotation_file)
        if not ann_path.is_file():
            raise DatasetNotFoundError(f"COCO annotation file not found: {ann_path}")

        try:
            with ann_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise MalformedDatasetError(
                f"Malformed JSON in COCO annotation file {ann_path.name}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise MalformedDatasetError(
                f"COCO annotation file must contain a root JSON object, got {type(data).__name__}"
            )

        required_keys = {"images", "annotations", "categories"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            raise MalformedDatasetError(
                f"COCO dataset missing required top-level keys: {sorted(missing_keys)}"
            )

        # 1. Parse categories
        categories: dict[int, str] = {}
        for cat in data["categories"]:
            if not isinstance(cat, dict) or "id" not in cat or "name" not in cat:
                raise MalformedDatasetError(
                    f"Malformed category record in COCO dataset: {cat!r}"
                )
            cat_id = cat["id"]
            if not isinstance(cat_id, int):
                raise MalformedDatasetError(
                    f"Category ID must be an integer, got {cat_id!r}"
                )
            categories[cat_id] = str(cat["name"])

        # 2. Resolve image directory
        resolved_img_dir: Optional[Path] = None
        if image_dir is not None:
            resolved_img_dir = Path(image_dir)
        else:
            candidates = [ann_path.parent / "images", ann_path.parent]
            for candidate in candidates:
                if candidate.is_dir():
                    resolved_img_dir = candidate
                    break

        # 3. Parse images
        images_map: dict[Union[int, str], DatasetImage] = {}

        for img in data["images"]:
            if not isinstance(img, dict):
                raise MalformedDatasetError(f"Malformed image record in COCO dataset: {img!r}")
            for req in ("id", "file_name", "width", "height"):
                if req not in img:
                    raise MalformedDatasetError(
                        f"Image record missing required field '{req}': {img!r}"
                    )

            img_id = img["id"]
            if img_id in images_map:
                raise MalformedDatasetError(f"Duplicate image id in COCO dataset: {img_id}")

            width = img["width"]
            height = img["height"]
            if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                raise MalformedDatasetError(
                    f"Image {img_id} width and height must be numeric, got {width!r}, {height!r}"
                )
            if width <= 0 or height <= 0:
                raise MalformedDatasetError(
                    f"Image {img_id} dimensions must be positive, got width={width}, height={height}"
                )

            file_name = str(img["file_name"])
            file_path = (resolved_img_dir / file_name) if resolved_img_dir is not None else None

            images_map[img_id] = DatasetImage(
                image_id=img_id,
                file_name=file_name,
                width=int(width),
                height=int(height),
                annotations=[],
                warnings=[],
                file_path=file_path,
            )

        # 4. Parse annotations
        for ann in data["annotations"]:
            if not isinstance(ann, dict):
                raise MalformedDatasetError(f"Malformed annotation record: {ann!r}")

            for req in ("id", "image_id", "category_id", "bbox"):
                if req not in ann:
                    raise MalformedDatasetError(
                        f"Annotation record missing required field '{req}': {ann!r}"
                    )

            ann_id = ann["id"]
            img_id = ann["image_id"]
            cat_id = ann["category_id"]

            if img_id not in images_map:
                raise InvalidAnnotationError(
                    f"Annotation {ann_id} references nonexistent image_id: {img_id}"
                )

            if cat_id not in categories:
                raise InvalidAnnotationError(
                    f"Annotation {ann_id} references nonexistent category_id: {cat_id}"
                )

            raw_bbox = ann["bbox"]
            if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
                raise InvalidBoundingBoxError(
                    f"Annotation {ann_id} bbox must have 4 elements [x, y, w, h], got: {raw_bbox!r}"
                )

            try:
                x, y, w, h = (float(v) for v in raw_bbox)
            except (ValueError, TypeError) as exc:
                raise InvalidBoundingBoxError(
                    f"Annotation {ann_id} bbox contains non-numeric values: {raw_bbox!r}"
                ) from exc

            # BoundingBox validation handles negative w/h and NaN/inf
            bbox = BoundingBox(xmin=x, ymin=y, width=w, height=h)

            target_image = images_map[img_id]
            ann_warnings: list[str] = []

            # Boundary validation
            is_out_of_bounds = (
                x < 0
                or y < 0
                or (x + w) > target_image.width
                or (y + h) > target_image.height
            )

            if is_out_of_bounds:
                warning_msg = (
                    f"Annotation {ann_id} bbox [{x}, {y}, {w}, {h}] extends outside "
                    f"image bounds ({target_image.width}x{target_image.height})"
                )
                if strict_bounds:
                    raise BoundingBoxOutOfBoundsError(warning_msg)
                ann_warnings.append(warning_msg)
                target_image.warnings.append(warning_msg)

            area = ann.get("area")
            if area is not None:
                try:
                    area = float(area)
                except (ValueError, TypeError):
                    area = float(w * h)
            else:
                area = float(w * h)

            iscrowd = ann.get("iscrowd")
            if iscrowd is not None:
                try:
                    iscrowd = int(iscrowd)
                except (ValueError, TypeError):
                    iscrowd = 0

            annotation_obj = Annotation(
                annotation_id=ann_id,
                category_id=cat_id,
                category_name=categories[cat_id],
                bbox=bbox,
                area=area,
                iscrowd=iscrowd,
                warnings=ann_warnings,
            )
            target_image.annotations.append(annotation_obj)

        name = dataset_name or ann_path.stem
        root = resolved_img_dir.parent if resolved_img_dir is not None else ann_path.parent

        return Dataset(
            name=name,
            format=DatasetFormat.COCO,
            images=list(images_map.values()),
            categories=categories,
            warnings=[],
            root_path=root,
        )
