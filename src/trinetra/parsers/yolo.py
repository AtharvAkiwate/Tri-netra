"""YOLO format dataset parser for TRI-NETRA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

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
from trinetra.utils.image_utils import get_image_dimensions

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IGNORED_LABEL_FILENAMES = {"classes.txt", "obj.names"}


class YoloParser:
    """Parser for YOLO TXT format datasets."""

    def parse(
        self,
        data_dir: Union[str, Path],
        images_dir: Optional[Union[str, Path]] = None,
        labels_dir: Optional[Union[str, Path]] = None,
        classes: Optional[Union[list[str], dict[int, str], str, Path]] = None,
        default_image_dims: Optional[tuple[int, int]] = None,
        dataset_name: Optional[str] = None,
        strict_bounds: bool = False,
        strict_orphans: Optional[bool] = None,
    ) -> Dataset:
        """Parse a YOLO dataset into the common dataset representation.

        Args:
            data_dir: Root dataset directory.
            images_dir: Optional override for images directory.
            labels_dir: Optional override for labels directory.
            classes: Class mapping (list of names, dict of id->name, or path to classes.txt / obj.names).
            default_image_dims: Optional fallback (width, height) in pixels if images are not readable.
            dataset_name: Optional human-readable name for the dataset.
            strict_bounds: If True, raises BoundingBoxOutOfBoundsError when a bounding box
                           extends outside [0, 1] normalized boundaries. If False (default),
                           preserves original coordinates without clipping and records a diagnostic warning.
            strict_orphans: If True, raises InvalidAnnotationError when a label file exists
                            without a corresponding image. If None, defaults to strict_bounds.

        Returns:
            Dataset: Standardized Common Dataset Representation.

        Raises:
            DatasetNotFoundError: If directories or required files are not found.
            MalformedDatasetError: If label files contain malformed lines or tokens.
            InvalidAnnotationError: If a class ID is invalid or an orphan label is found in strict mode.
            InvalidBoundingBoxError: If box width/height is negative or coordinates are non-numeric/NaN.
            BoundingBoxOutOfBoundsError: If strict_bounds=True and a bounding box exceeds image dimensions.
        """
        root_path = Path(data_dir)
        if not root_path.exists():
            raise DatasetNotFoundError(f"YOLO dataset directory not found: {root_path}")

        # 1. Resolve images directory
        resolved_images_dir: Path
        if images_dir is not None:
            resolved_images_dir = Path(images_dir)
        elif (root_path / "images").is_dir():
            resolved_images_dir = root_path / "images"
        elif root_path.is_dir():
            resolved_images_dir = root_path
        else:
            raise DatasetNotFoundError(f"Cannot resolve images directory for: {root_path}")

        if not resolved_images_dir.exists():
            raise DatasetNotFoundError(f"YOLO images directory not found: {resolved_images_dir}")

        # 2. Resolve labels directory
        resolved_labels_dir: Path
        if labels_dir is not None:
            resolved_labels_dir = Path(labels_dir)
        elif (root_path / "labels").is_dir():
            resolved_labels_dir = root_path / "labels"
        else:
            resolved_labels_dir = resolved_images_dir

        if not resolved_labels_dir.exists():
            raise DatasetNotFoundError(f"YOLO labels directory not found: {resolved_labels_dir}")

        # 3. Resolve class definitions
        categories = self._resolve_categories(root_path, resolved_labels_dir, classes)

        # 4. Discover image files
        image_files = [
            p for p in resolved_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        image_files.sort(key=lambda p: p.name)

        if not image_files:
            raise DatasetNotFoundError(
                f"No supported image files ({sorted(SUPPORTED_IMAGE_EXTENSIONS)}) found in {resolved_images_dir}"
            )

        dataset_warnings: list[str] = []
        is_strict_orphans = strict_bounds if strict_orphans is None else strict_orphans

        # 5. Check for orphan label files (labels with no corresponding image)
        discovered_image_stems = {p.stem for p in image_files}
        if resolved_labels_dir.is_dir():
            for label_candidate in resolved_labels_dir.iterdir():
                if (
                    label_candidate.is_file()
                    and label_candidate.suffix.lower() == ".txt"
                    and label_candidate.name.lower() not in IGNORED_LABEL_FILENAMES
                ):
                    if label_candidate.stem not in discovered_image_stems:
                        orphan_msg = f"Orphan label file without corresponding image: {label_candidate.name}"
                        if is_strict_orphans:
                            raise InvalidAnnotationError(orphan_msg)
                        dataset_warnings.append(orphan_msg)

        # 6. Parse each image and its corresponding label file
        dataset_images: list[DatasetImage] = []
        annotation_counter = 1
        total_annotations_found = 0

        for img_path in image_files:
            # Determine dimensions
            width, height = self._get_dimensions(img_path, default_image_dims)

            image_warnings: list[str] = []
            annotations: list[Annotation] = []

            # Check for label file
            label_file = resolved_labels_dir / f"{img_path.stem}.txt"
            if not label_file.is_file():
                # Check adjacent to image if different directory
                adjacent_label = img_path.with_suffix(".txt")
                if adjacent_label.is_file():
                    label_file = adjacent_label

            if label_file.is_file():
                try:
                    with label_file.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                except UnicodeDecodeError as exc:
                    raise MalformedDatasetError(
                        f"Unable to read label file {label_file.name} as UTF-8: {exc}"
                    ) from exc

                for line_idx, raw_line in enumerate(lines, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    tokens = line.split()
                    if len(tokens) != 5:
                        raise MalformedDatasetError(
                            f"Malformed YOLO line in {label_file.name}:{line_idx}: "
                            f"expected 5 tokens (<class_id> <cx> <cy> <w> <h>), got {len(tokens)}: {raw_line!r}"
                        )

                    try:
                        class_id = int(tokens[0])
                    except ValueError as exc:
                        raise MalformedDatasetError(
                            f"Non-integer class ID '{tokens[0]}' in {label_file.name}:{line_idx}"
                        ) from exc

                    try:
                        cx = float(tokens[1])
                        cy = float(tokens[2])
                        w = float(tokens[3])
                        h = float(tokens[4])
                    except ValueError as exc:
                        raise MalformedDatasetError(
                            f"Non-numeric coordinates in {label_file.name}:{line_idx}: {raw_line!r}"
                        ) from exc

                    for coord_val, coord_name in [(cx, "cx"), (cy, "cy"), (w, "w"), (h, "h")]:
                        if math.isnan(coord_val) or math.isinf(coord_val):
                            raise InvalidBoundingBoxError(
                                f"Coordinate '{coord_name}' is not finite in {label_file.name}:{line_idx}: {coord_val!r}"
                            )

                    if class_id < 0:
                        raise InvalidAnnotationError(
                            f"Invalid negative class ID {class_id} in {label_file.name}:{line_idx}"
                        )

                    if categories and class_id not in categories:
                        raise InvalidAnnotationError(
                            f"Class ID {class_id} in {label_file.name}:{line_idx} not found in known categories: {sorted(categories.keys())}"
                        )

                    if w <= 0.0 or h <= 0.0:
                        raise InvalidBoundingBoxError(
                            f"Non-positive bounding box width/height (w={w}, h={h}) in {label_file.name}:{line_idx}"
                        )

                    ann_warnings: list[str] = []
                    # Check if normalized coordinates spill outside [0.0, 1.0]
                    norm_xmin = cx - w / 2.0
                    norm_ymin = cy - h / 2.0
                    norm_xmax = cx + w / 2.0
                    norm_ymax = cy + h / 2.0

                    is_out_of_bounds = (
                        norm_xmin < 0.0
                        or norm_ymin < 0.0
                        or norm_xmax > 1.0
                        or norm_ymax > 1.0
                    )

                    if is_out_of_bounds:
                        warning_msg = (
                            f"YOLO bbox in {label_file.name}:{line_idx} extends outside normalized [0, 1] range: "
                            f"[cx={cx}, cy={cy}, w={w}, h={h}] -> [xmin={norm_xmin:.4f}, ymin={norm_ymin:.4f}, xmax={norm_xmax:.4f}, ymax={norm_ymax:.4f}]"
                        )
                        if strict_bounds:
                            raise BoundingBoxOutOfBoundsError(warning_msg)
                        ann_warnings.append(warning_msg)
                        image_warnings.append(warning_msg)

                    # Convert to canonical absolute pixel BoundingBox
                    # Preserving original normalized coordinates in bbox.normalized
                    bbox = BoundingBox.from_yolo(
                        cx=cx,
                        cy=cy,
                        w=w,
                        h=h,
                        img_w=width,
                        img_h=height,
                    )

                    category_name = categories.get(class_id)

                    annotation = Annotation(
                        annotation_id=annotation_counter,
                        category_id=class_id,
                        category_name=category_name,
                        bbox=bbox,
                        area=bbox.width * bbox.height,
                        iscrowd=0,
                        warnings=ann_warnings,
                    )
                    annotations.append(annotation)
                    annotation_counter += 1
                    total_annotations_found += 1

            dataset_images.append(
                DatasetImage(
                    image_id=img_path.stem,
                    file_name=img_path.name,
                    width=width,
                    height=height,
                    annotations=annotations,
                    warnings=image_warnings,
                    file_path=img_path,
                )
            )

        name = dataset_name or root_path.name

        # 7. Warn if annotations exist but no class definitions were found
        if not categories and total_annotations_found > 0:
            dataset_warnings.append(
                f"No class mapping found (classes.txt / obj.names / classes argument). "
                f"Category names are unavailable for dataset '{name}'."
            )

        return Dataset(
            name=name,
            format=DatasetFormat.YOLO,
            images=dataset_images,
            categories=categories,
            warnings=dataset_warnings,
            root_path=root_path,
        )

    def _resolve_categories(
        self,
        root_path: Path,
        labels_dir: Path,
        classes: Optional[Union[list[str], dict[int, str], str, Path]],
    ) -> dict[int, str]:
        """Resolve category mapping from explicit input or standard class files."""
        if classes is not None:
            if isinstance(classes, list):
                return {idx: str(name) for idx, name in enumerate(classes)}
            if isinstance(classes, dict):
                return {int(k): str(v) for k, v in classes.items()}
            if isinstance(classes, (str, Path)):
                class_file = Path(classes)
                if not class_file.is_file():
                    raise DatasetNotFoundError(f"Classes file not found: {class_file}")
                return self._read_classes_file(class_file)
            raise ValueError(f"Unsupported type for classes: {type(classes).__name__}")

        # Search for classes.txt or obj.names in standard locations
        candidates = [
            root_path / "classes.txt",
            root_path / "obj.names",
            labels_dir / "classes.txt",
            labels_dir / "obj.names",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return self._read_classes_file(candidate)

        return {}

    def _read_classes_file(self, file_path: Path) -> dict[int, str]:
        """Read newline-delimited class names from a text file."""
        categories: dict[int, str] = {}
        with file_path.open("r", encoding="utf-8") as f:
            for idx, raw_line in enumerate(f):
                line = raw_line.strip()
                if line:
                    categories[idx] = line
        return categories

    def _get_dimensions(
        self,
        img_path: Path,
        default_dims: Optional[tuple[int, int]],
    ) -> tuple[int, int]:
        """Extract image dimensions from header, or use default fallback if provided."""
        try:
            return get_image_dimensions(img_path)
        except Exception as exc:
            if default_dims is not None:
                if (
                    isinstance(default_dims, tuple)
                    and len(default_dims) == 2
                    and default_dims[0] > 0
                    and default_dims[1] > 0
                ):
                    return (int(default_dims[0]), int(default_dims[1]))
                raise MalformedDatasetError(
                    f"Invalid default_image_dims: {default_dims!r}"
                )
            raise MalformedDatasetError(
                f"Could not read image dimensions from {img_path.name}: {exc}. "
                "Specify default_image_dims=(width, height) if images cannot be read directly."
            ) from exc
