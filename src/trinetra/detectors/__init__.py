"""Layer 1C detector APIs."""

from trinetra.detectors.near_duplicate import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DuplicateCluster,
    DuplicateMember,
    DuplicatePair,
    EmbeddingSpaceMismatchError,
    InvalidFeatureBatchError,
    NearDuplicateDetector,
    NearDuplicateError,
    NearDuplicateResult,
    cosine_similarity,
)

__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "DuplicateCluster",
    "DuplicateMember",
    "DuplicatePair",
    "EmbeddingSpaceMismatchError",
    "InvalidFeatureBatchError",
    "NearDuplicateDetector",
    "NearDuplicateError",
    "NearDuplicateResult",
    "cosine_similarity",
]
