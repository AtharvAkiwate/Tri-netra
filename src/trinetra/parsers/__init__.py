"""TRI-NETRA Dataset Parsers.

Supports loading COCO JSON and YOLO TXT datasets into a standardized Common Dataset Representation.
"""

from trinetra.parsers.coco import CocoParser
from trinetra.parsers.exceptions import (
    BoundingBoxOutOfBoundsError,
    DatasetNotFoundError,
    InvalidAnnotationError,
    InvalidBoundingBoxError,
    MalformedDatasetError,
    TrinetraParserError,
)
from trinetra.parsers.factory import load_dataset
from trinetra.parsers.models import (
    Annotation,
    BoundingBox,
    Dataset,
    DatasetFormat,
    DatasetImage,
)
from trinetra.parsers.yolo import YoloParser

__all__ = [
    "Annotation",
    "BoundingBox",
    "BoundingBoxOutOfBoundsError",
    "CocoParser",
    "Dataset",
    "DatasetFormat",
    "DatasetImage",
    "DatasetNotFoundError",
    "InvalidAnnotationError",
    "InvalidBoundingBoxError",
    "MalformedDatasetError",
    "TrinetraParserError",
    "YoloParser",
    "load_dataset",
]
