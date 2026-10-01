"""Pillow-based RGB image loading for TRI-NETRA feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from trinetra.features.exceptions import BackendUnavailableError, ImageUnreadableError

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError as exc:  # pragma: no cover - exercised when Pillow is absent
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    UnidentifiedImageError = Exception  # type: ignore[misc, assignment]
    _PILLOW_IMPORT_ERROR = exc
else:
    _PILLOW_IMPORT_ERROR = None


def _require_pillow() -> None:
    if Image is None:
        raise BackendUnavailableError(
            "Pillow is required for image loading. Install with: pip install 'trinetra[features]'"
        ) from _PILLOW_IMPORT_ERROR


def load_rgb_image(file_path: Union[str, Path]) -> Image.Image:
    """Decode an image file as an in-memory RGB PIL Image.

    Applies EXIF orientation when present. Does not invent pixels for missing
    or corrupt files.

    Args:
        file_path: Path to a raster image on disk.

    Returns:
        RGB PIL Image (copied off the file handle).

    Raises:
        BackendUnavailableError: If Pillow is not installed.
        ImageUnreadableError: If the path is missing or the file cannot be decoded.
    """
    _require_pillow()
    path = Path(file_path)
    if not path.is_file():
        raise ImageUnreadableError(f"Image file not found: {path}", code="image_not_found")

    try:
        with Image.open(path) as opened:
            transposed = ImageOps.exif_transpose(opened)
            rgb = transposed.convert("RGB")
            rgb.load()
            if rgb.width <= 0 or rgb.height <= 0:
                raise ImageUnreadableError(
                    f"Image has invalid dimensions ({rgb.width}x{rgb.height}): {path}",
                    code="invalid_image_dimensions",
                )
            return rgb.copy()
    except ImageUnreadableError:
        raise
    except UnidentifiedImageError as exc:
        raise ImageUnreadableError(
            f"Unrecognized or unsupported image format: {path}", code="unrecognized_image"
        ) from exc
    except OSError as exc:
        raise ImageUnreadableError(f"Could not decode image {path}: {exc}", code="image_decode_error") from exc


def try_load_rgb_image(file_path: Optional[Union[str, Path]]) -> tuple[Optional[Image.Image], Optional[str]]:
    """Attempt to load an RGB image without inventing a raster.

    Args:
        file_path: Path to load, or None when the dataset record has no path.

    Returns:
        (image, None) on success, or (None, reason) when the image cannot be used.
    """
    if file_path is None:
        return None, "No file_path on dataset image"
    try:
        return load_rgb_image(file_path), None
    except ImageUnreadableError as exc:
        return None, str(exc)
