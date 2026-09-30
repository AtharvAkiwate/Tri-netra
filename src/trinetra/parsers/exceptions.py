"""Custom exceptions for TRI-NETRA dataset parsers."""


class TrinetraParserError(Exception):
    """Base exception for all TRI-NETRA dataset parser errors."""
    pass


class DatasetNotFoundError(TrinetraParserError):
    """Raised when a dataset file, directory, or required dependency path is not found."""
    pass


class MalformedDatasetError(TrinetraParserError):
    """Raised when a dataset file or structure is malformed or unparseable."""
    pass


class InvalidAnnotationError(TrinetraParserError):
    """Raised when an annotation contains invalid references or attributes (e.g. unknown image or category ID)."""
    pass


class InvalidBoundingBoxError(TrinetraParserError):
    """Raised when a bounding box contains fatal defects (negative width/height, non-numeric or NaN values)."""
    pass


class BoundingBoxOutOfBoundsError(InvalidBoundingBoxError):
    """Raised when strict boundary validation is enabled and a bounding box extends beyond image boundaries."""
    pass
