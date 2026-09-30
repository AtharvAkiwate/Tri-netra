"""Unified dataset loading interface for TRI-NETRA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from trinetra.parsers.coco import CocoParser
from trinetra.parsers.exceptions import (
    DatasetNotFoundError,
    TrinetraParserError,
)
from trinetra.parsers.models import Dataset, DatasetFormat
from trinetra.parsers.yolo import YoloParser


def load_dataset(
    source: Union[str, Path],
    format: Optional[Union[str, DatasetFormat]] = None,
    **kwargs: Any,
) -> Dataset:
    """Load a dataset into the common TRI-NETRA representation.

    Auto-detects whether the source is a COCO JSON file or a YOLO TXT dataset directory
    if the format is not explicitly provided.

    Args:
        source: Path to the dataset annotation file or directory.
        format: Optional explicit format ("coco" or "yolo" or DatasetFormat enum).
        **kwargs: Additional format-specific arguments passed to the underlying parser:
            For COCO:
                image_dir: Optional path to images directory.
                strict_bounds: Whether to raise error on out-of-bounds bounding boxes (default: False).
            For YOLO:
                images_dir: Optional path to images directory.
                labels_dir: Optional path to labels directory.
                classes: Class mapping (list, dict, or path to classes.txt).
                default_image_dims: Fallback (width, height) tuple if image dimensions cannot be read.
                strict_bounds: Whether to raise error on out-of-bounds bounding boxes (default: False).
                strict_orphans: Whether to raise error on orphan label files (default: strict_bounds).

    Returns:
        Dataset: Standardized Common Dataset Representation.

    Raises:
        DatasetNotFoundError: If the source path does not exist.
        TrinetraParserError: If format cannot be determined, is unsupported, or has ambiguous files.
    """
    path = Path(source)
    if not path.exists():
        raise DatasetNotFoundError(f"Dataset source path does not exist: {path}")

    # Determine format
    resolved_format: str
    if format is not None:
        if isinstance(format, DatasetFormat):
            resolved_format = format.value
        else:
            resolved_format = str(format).strip().lower()
    else:
        # Auto-detect format
        if path.is_file() and path.suffix.lower() == ".json":
            resolved_format = DatasetFormat.COCO.value
        elif path.is_dir():
            has_labels = (path / "labels").is_dir()
            has_images = (path / "images").is_dir()
            has_classes = (path / "classes.txt").is_file() or (path / "obj.names").is_file()
            # Check if there is a single json file in the directory
            json_files = list(path.glob("*.json"))

            if has_labels or has_images or has_classes:
                resolved_format = DatasetFormat.YOLO.value
            elif len(json_files) == 1 and not has_labels:
                # Folder with a single COCO JSON file
                return CocoParser().parse(json_files[0], image_dir=path, **kwargs)
            else:
                # Default attempt as YOLO directory
                resolved_format = DatasetFormat.YOLO.value
        else:
            raise TrinetraParserError(
                f"Cannot automatically determine dataset format for {path}. "
                "Specify format='coco' or format='yolo' explicitly."
            )

    if resolved_format == DatasetFormat.COCO.value:
        if path.is_dir():
            json_files = list(path.glob("*.json"))
            if len(json_files) == 1:
                return CocoParser().parse(json_files[0], image_dir=path, **kwargs)
            elif len(json_files) == 0:
                raise DatasetNotFoundError(f"No COCO JSON annotation files found in directory: {path}")
            else:
                file_names = sorted(f.name for f in json_files)
                raise TrinetraParserError(
                    f"Ambiguous COCO annotation files in {path}: {file_names}. "
                    "Specify the exact JSON annotation file path directly."
                )
        return CocoParser().parse(path, **kwargs)
    elif resolved_format == DatasetFormat.YOLO.value:
        return YoloParser().parse(path, **kwargs)
    else:
        raise TrinetraParserError(
            f"Unsupported dataset format '{format}'. Supported formats: 'coco', 'yolo'"
        )
