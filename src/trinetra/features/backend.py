"""Feature extractor backend protocol for TRI-NETRA Layer 1B."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from trinetra.features.models import FeatureBatch
from trinetra.parsers.models import Dataset


@runtime_checkable
class FeatureExtractor(Protocol):
    """Extract image embeddings from a Layer 1A Dataset.

    Implementations must not invent vectors for failed images. Individual image
    decode failures belong in FeatureBatch.failures. Missing weights and an
    unavailable backend/dependency must propagate as package-level exceptions.
    """

    def extract(self, dataset: Dataset) -> FeatureBatch:
        """Produce embeddings for readable images in ``dataset``.

        Args:
            dataset: Normalized Layer 1A dataset.

        Returns:
            FeatureBatch containing only successfully computed embeddings.

        Raises:
            WeightsNotFoundError: Required local weights are unavailable.
            BackendUnavailableError: A required backend or dependency is unavailable.
        """
        ...
