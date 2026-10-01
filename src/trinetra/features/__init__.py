"""TRI-NETRA Layer 1B feature extraction.

Produces image embeddings from a Layer 1A Dataset. Detection and findings are
out of scope for this module.
"""

from trinetra.features.backend import FeatureExtractor
from trinetra.features.cache import compute_cache_key, fingerprint_dataset
from trinetra.features.dummy import DummyFeatureExtractor
from trinetra.features.dinov2 import DINOv2FeatureExtractor, DinoV2FeatureExtractor
from trinetra.features.exceptions import (
    BackendUnavailableError,
    EmbeddingStoreError,
    ImageUnreadableError,
    FeatureExtractionError,
    IncompatibleWeightsError,
    TrinetraFeatureError,
    WeightsNotFoundError,
)
from trinetra.features.resnet50 import ResNet50FeatureExtractor
from trinetra.features.runner import FeatureExtractionRunner, Layer1BFeatureExtractor
from trinetra.features.image_io import load_rgb_image, try_load_rgb_image
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure, make_embedding_space_id
from trinetra.features.store import EmbeddingStore

__all__ = [
    "BackendUnavailableError",
    "compute_cache_key",
    "DummyFeatureExtractor",
    "DINOv2FeatureExtractor",
    "DinoV2FeatureExtractor",
    "EmbeddingStore",
    "EmbeddingStoreError",
    "FeatureBatch",
    "FeatureExtractionError",
    "FeatureExtractionRunner",
    "FeatureExtractor",
    "ImageEmbedding",
    "ImageFailure",
    "ImageUnreadableError",
    "IncompatibleWeightsError",
    "TrinetraFeatureError",
    "WeightsNotFoundError",
    "ResNet50FeatureExtractor",
    "Layer1BFeatureExtractor",
    "load_rgb_image",
    "make_embedding_space_id",
    "fingerprint_dataset",
    "try_load_rgb_image",
]
