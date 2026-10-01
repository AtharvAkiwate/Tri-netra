"""Validated, serializable feature extraction data models."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from trinetra.features.exceptions import TrinetraFeatureError
from trinetra.features.cache import compute_cache_key

ImageId = Union[int, str]
FEATURE_BATCH_SCHEMA_VERSION = 3


def _as_float32_vector(vector: np.ndarray, embedding_dim: Optional[int] = None) -> np.ndarray:
    if not isinstance(vector, np.ndarray) or vector.ndim != 1 or vector.size == 0:
        raise TrinetraFeatureError("Embedding vector must be a non-empty 1-D numpy array")
    if embedding_dim is not None and vector.size != embedding_dim:
        raise TrinetraFeatureError(
            f"Embedding vector length {vector.size} does not match embedding_dim {embedding_dim}"
        )
    if not np.issubdtype(vector.dtype, np.floating):
        raise TrinetraFeatureError("Embedding vector dtype must be floating-point")
    if not np.isfinite(vector).all():
        raise TrinetraFeatureError("Embedding vector contains NaN or infinite values")
    return np.ascontiguousarray(vector, dtype=np.float32)


def _canonical_config(value: dict[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrinetraFeatureError(f"{label} must be a JSON-compatible object")
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TrinetraFeatureError(f"{label} must contain JSON-compatible values: {exc}") from exc


def make_embedding_space_id(
    *, model_id: str, weights_identity: str, preprocessing: dict[str, Any],
    embedding_dim: int, normalization: dict[str, Any],
) -> str:
    """Return a stable identifier for all settings that define vector comparability."""
    payload = {
        "model_id": model_id,
        "weights_identity": weights_identity,
        "preprocessing": _canonical_config(preprocessing, "preprocessing"),
        "embedding_dim": embedding_dim,
        "normalization": _canonical_config(normalization, "normalization"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return f"sha256:{digest}"


@dataclass
class ImageEmbedding:
    """A vector for one successfully processed Layer 1A image."""

    image_id: ImageId
    file_name: str
    vector: np.ndarray
    file_path: Optional[Path] = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.image_id, bool) or not isinstance(self.image_id, (int, str)):
            raise TrinetraFeatureError("image_id must be an int or str")
        if not isinstance(self.file_name, str) or not self.file_name:
            raise TrinetraFeatureError("file_name must be a non-empty string")
        self.vector = _as_float32_vector(self.vector)
        if self.file_path is not None and not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)


@dataclass
class ImageFailure:
    """A failed per-image extraction, kept separate from successful vectors."""

    image_id: ImageId
    file_path: Optional[Path]
    failure_code: str
    reason: str

    def __post_init__(self) -> None:
        if isinstance(self.image_id, bool) or not isinstance(self.image_id, (int, str)):
            raise TrinetraFeatureError("failure image_id must be an int or str")
        if self.file_path is not None and not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
        if not isinstance(self.failure_code, str) or not self.failure_code.strip():
            raise TrinetraFeatureError("failure_code must be a non-empty string")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise TrinetraFeatureError("failure reason must be a non-empty string")


@dataclass
class FeatureBatch:
    """Successful embeddings and their reproducibility/failure metadata."""

    dataset_name: str
    backend: str
    model_id: str
    embedding_dim: int
    device: str
    embeddings: list[ImageEmbedding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    normalization: dict[str, Any] = field(default_factory=dict)
    weights_identity: str = "unspecified"
    failures: list[ImageFailure] = field(default_factory=list)
    embedding_space_id: Optional[str] = None
    schema_version: int = FEATURE_BATCH_SCHEMA_VERSION
    requested_extractor: Optional[str] = None
    actual_extractor: Optional[str] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    input_fingerprint: Optional[str] = None
    cache_key: Optional[str] = None

    def __post_init__(self) -> None:
        for label in ("dataset_name", "backend", "model_id", "device", "weights_identity"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise TrinetraFeatureError(f"{label} must be a non-empty string")
        if isinstance(self.embedding_dim, bool) or not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise TrinetraFeatureError(f"embedding_dim must be a positive integer, got {self.embedding_dim!r}")
        if type(self.schema_version) is not int or self.schema_version != FEATURE_BATCH_SCHEMA_VERSION:
            raise TrinetraFeatureError(
                f"Unsupported FeatureBatch schema_version {self.schema_version!r}; "
                f"expected {FEATURE_BATCH_SCHEMA_VERSION}"
            )
        if self.requested_extractor is None:
            self.requested_extractor = self.backend
        if self.actual_extractor is None:
            self.actual_extractor = self.backend
        if not isinstance(self.requested_extractor, str) or not self.requested_extractor.strip():
            raise TrinetraFeatureError("requested_extractor must be a non-empty string")
        if not isinstance(self.actual_extractor, str) or not self.actual_extractor.strip():
            raise TrinetraFeatureError("actual_extractor must be a non-empty string")
        if self.actual_extractor != self.backend:
            raise TrinetraFeatureError("actual_extractor must match the FeatureBatch backend")
        if type(self.fallback_used) is not bool:
            raise TrinetraFeatureError("fallback_used must be a bool")
        if self.fallback_used:
            if self.requested_extractor == self.actual_extractor:
                raise TrinetraFeatureError("fallback must use a different actual_extractor")
            if not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip():
                raise TrinetraFeatureError("fallback_reason is required when fallback_used is true")
        elif self.requested_extractor != self.actual_extractor or self.fallback_reason is not None:
            raise TrinetraFeatureError("non-fallback provenance must have matching extractors and no reason")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str) or not self.input_fingerprint.startswith("sha256:")
        ):
            raise TrinetraFeatureError("input_fingerprint must be a SHA-256 identity")
        self.preprocessing = _canonical_config(self.preprocessing, "preprocessing")
        self.normalization = _canonical_config(self.normalization, "normalization")
        if not self.preprocessing:
            raise TrinetraFeatureError("preprocessing metadata must not be empty")
        if not self.normalization:
            raise TrinetraFeatureError("normalization metadata must not be empty")
        seen: set[ImageId] = set()
        for item in self.embeddings:
            if not isinstance(item, ImageEmbedding):
                raise TrinetraFeatureError("embeddings must contain ImageEmbedding records")
            if item.image_id in seen:
                raise TrinetraFeatureError(f"Duplicate image_id in embeddings: {item.image_id!r}")
            seen.add(item.image_id)
            item.vector = _as_float32_vector(item.vector, embedding_dim=self.embedding_dim)
        failed_ids: set[ImageId] = set()
        for failure in self.failures:
            if not isinstance(failure, ImageFailure):
                raise TrinetraFeatureError("failures must contain ImageFailure records")
            if failure.image_id in seen:
                raise TrinetraFeatureError(f"image_id cannot have both an embedding and failure: {failure.image_id!r}")
            if failure.image_id in failed_ids:
                raise TrinetraFeatureError(f"Duplicate failed image_id: {failure.image_id!r}")
            failed_ids.add(failure.image_id)
        expected = make_embedding_space_id(
            model_id=self.model_id,
            weights_identity=self.weights_identity,
            preprocessing=self.preprocessing,
            embedding_dim=self.embedding_dim,
            normalization=self.normalization,
        )
        if self.embedding_space_id is not None and self.embedding_space_id != expected:
            raise TrinetraFeatureError("embedding_space_id does not match model/weights/preprocessing/dimension/normalization")
        self.embedding_space_id = expected
        expected_cache_key = self.make_cache_key()
        if self.cache_key is not None and self.cache_key != expected_cache_key:
            raise TrinetraFeatureError("cache_key does not match batch input and extractor metadata")
        self.cache_key = expected_cache_key

    def make_cache_key(self) -> Optional[str]:
        """Hash source bytes and extraction identity; ``None`` means not cacheable."""
        if self.input_fingerprint is None:
            return None
        return compute_cache_key(
            schema_version=self.schema_version,
            input_fingerprint=self.input_fingerprint,
            embedding_space_id=self.embedding_space_id or "",
            requested_extractor=self.requested_extractor or self.backend,
            actual_extractor=self.actual_extractor or self.backend,
            fallback_used=self.fallback_used,
        )

    @property
    def embedding_count(self) -> int:
        return len(self.embeddings)

    def as_matrix(self) -> np.ndarray:
        if not self.embeddings:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return np.stack([item.vector for item in self.embeddings], axis=0)
