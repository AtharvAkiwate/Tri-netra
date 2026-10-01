"""Custom exceptions for TRI-NETRA feature extraction."""


class TrinetraFeatureError(Exception):
    """Base exception for all TRI-NETRA feature extraction errors."""


class BackendUnavailableError(TrinetraFeatureError):
    """Raised when a required feature-extraction backend or dependency is unavailable."""


class WeightsNotFoundError(TrinetraFeatureError):
    """Raised when local model weights are required but missing or unreadable."""


class IncompatibleWeightsError(TrinetraFeatureError):
    """Raised when present model weights/configuration do not match the backend."""


class FeatureExtractionError(TrinetraFeatureError):
    """Raised when a complete extractor run fails or required fallback is unavailable."""


class ImageUnreadableError(TrinetraFeatureError):
    """Raised when an image file cannot be decoded into a raster."""

    def __init__(self, message: str, code: str = "image_unreadable") -> None:
        super().__init__(message)
        self.code = code


class EmbeddingStoreError(TrinetraFeatureError):
    """Raised when embedding persistence or load fails."""
