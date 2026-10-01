"""Deterministic dummy feature extractor for Layer 1B Slice 1.

This backend exists so extraction, image I/O, and persistence can be tested
without neural-network weights. It is not an assurance detector.
"""

from __future__ import annotations

import numpy as np

from trinetra.features.cache import fingerprint_dataset
from trinetra.features.exceptions import ImageUnreadableError
from trinetra.features.image_io import load_rgb_image
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure
from trinetra.parsers.models import Dataset

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


class DummyFeatureExtractor:
    """Image-level extractor using a fixed, local, non-learned projection.

    Unreadable or missing images are skipped and reported in batch warnings.
    """

    backend = "dummy"
    model_id = "dummy-meanpool-v1"

    def __init__(self, embedding_dim: int = 8, device: str = "cpu") -> None:
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        self.embedding_dim = embedding_dim
        self.device = device

    def extract(self, dataset: Dataset) -> FeatureBatch:
        embeddings: list[ImageEmbedding] = []
        failures: list[ImageFailure] = []

        for image in dataset.images:
            try:
                if image.file_path is None:
                    raise ImageUnreadableError("No file_path on dataset image", code="image_path_missing")
                rgb = load_rgb_image(image.file_path)
            except ImageUnreadableError as exc:
                failures.append(
                    ImageFailure(
                        image_id=image.image_id,
                        file_path=image.file_path,
                        failure_code=exc.code,
                        reason=str(exc),
                    )
                )
                continue

            vector = self._embed(rgb)
            embeddings.append(
                ImageEmbedding(
                    image_id=image.image_id,
                    file_name=image.file_name,
                    vector=vector,
                    file_path=image.file_path,
                    warnings=[],
                )
            )

        return FeatureBatch(
            dataset_name=dataset.name,
            backend=self.backend,
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            device=self.device,
            embeddings=embeddings,
            failures=failures,
            input_fingerprint=fingerprint_dataset(dataset),
            preprocessing={
                "loader": "pillow-rgb-exif-transpose-v1",
                "resize": {"width": 32, "height": 32, "resample": "bilinear"},
                "input_range": [0.0, 1.0],
                "pooling": "rgb-channel-mean-repeat-v1",
            },
            normalization={"method": "l2", "zero_vector": "unchanged"},
            weights_identity="none",
        )

    def _embed(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((32, 32), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        channel_means = arr.mean(axis=(0, 1))
        vector = np.resize(channel_means, self.embedding_dim).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return np.ascontiguousarray(vector, dtype=np.float32)
