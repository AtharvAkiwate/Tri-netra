"""Tests for Pillow RGB image loading."""

from pathlib import Path

import pytest
from PIL import Image

from trinetra.features.exceptions import BackendUnavailableError, ImageUnreadableError
from trinetra.features.image_io import load_rgb_image, try_load_rgb_image


def test_load_rgb_image_success(tmp_path: Path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (12, 8), color=(10, 20, 30)).save(path)

    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.size == (12, 8)
    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_load_rgb_converts_non_rgb_mode(tmp_path: Path):
    path = tmp_path / "gray.png"
    Image.new("L", (4, 4), color=128).save(path)

    image = load_rgb_image(path)
    assert image.mode == "RGB"
    assert image.size == (4, 4)


def test_load_rgb_missing_file(tmp_path: Path):
    with pytest.raises(ImageUnreadableError, match="Image file not found"):
        load_rgb_image(tmp_path / "missing.png")


def test_load_rgb_corrupt_file(tmp_path: Path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not-an-image")

    with pytest.raises(ImageUnreadableError, match="Unrecognized or unsupported image format"):
        load_rgb_image(path)


def test_try_load_rgb_image_none_and_failure(tmp_path: Path):
    image, reason = try_load_rgb_image(None)
    assert image is None
    assert reason == "No file_path on dataset image"

    missing = tmp_path / "nope.jpg"
    image, reason = try_load_rgb_image(missing)
    assert image is None
    assert reason is not None
    assert "not found" in reason

    path = tmp_path / "ok.png"
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(path)
    image, reason = try_load_rgb_image(path)
    assert reason is None
    assert image is not None
    assert image.size == (2, 2)


def test_missing_pillow_is_not_reported_as_bad_image(monkeypatch, tmp_path: Path):
    import trinetra.features.image_io as image_io

    monkeypatch.setattr(image_io, "Image", None)
    with pytest.raises(BackendUnavailableError, match="Pillow is required"):
        try_load_rgb_image(tmp_path / "image.png")
