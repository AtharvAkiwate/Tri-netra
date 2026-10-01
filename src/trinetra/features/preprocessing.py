"""Shared deterministic preprocessing for learned image backends."""

from __future__ import annotations

from typing import Any

import numpy as np

from trinetra.features import image_io
from trinetra.features.exceptions import BackendUnavailableError, ImageUnreadableError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SHORTEST_EDGE = 256
CENTER_CROP_SIZE = 224


def preprocess_rgb_image(rgb_image: Any) -> np.ndarray:
    """Pillow RGB → shortest edge 256 bicubic → center crop 224 → ImageNet NCHW."""
    if image_io.Image is None:
        raise BackendUnavailableError("Pillow is required for feature preprocessing")
    width, height = rgb_image.size
    if width <= 0 or height <= 0:
        raise ImageUnreadableError(f"Image has invalid dimensions ({width}x{height})")
    if width <= height:
        resized_width = RESIZE_SHORTEST_EDGE
        resized_height = int(RESIZE_SHORTEST_EDGE * height / width)
    else:
        resized_height = RESIZE_SHORTEST_EDGE
        resized_width = int(RESIZE_SHORTEST_EDGE * width / height)
    resized = rgb_image.resize(
        (resized_width, resized_height), resample=image_io.Image.Resampling.BICUBIC
    )
    left = int(round((resized_width - CENTER_CROP_SIZE) / 2.0))
    top = int(round((resized_height - CENTER_CROP_SIZE) / 2.0))
    crop = resized.crop((left, top, left + CENTER_CROP_SIZE, top + CENTER_CROP_SIZE))
    pixels = np.asarray(crop, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    return np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1), dtype=np.float32)


def preprocessing_metadata() -> dict[str, Any]:
    return {
        "loader": "pillow-rgb-exif-transpose-v1",
        "resize": {"shortest_edge": RESIZE_SHORTEST_EDGE, "resample": "bicubic"},
        "center_crop": {"height": CENTER_CROP_SIZE, "width": CENTER_CROP_SIZE, "rounding": "nearest-even"},
        "color": "RGB",
        "scale": "uint8-div-255",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "layout": "NCHW",
        "interpolation_implementation": "Pillow",
        "augmentation": "none",
    }
