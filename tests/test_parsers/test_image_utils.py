"""Tests for pure-Python image header dimension reader."""

import struct
from pathlib import Path

import pytest

from trinetra.utils.image_utils import get_image_dimensions


def test_png_dimensions(tmp_path: Path):
    png_file = tmp_path / "sample.png"
    # PNG signature (8 bytes) + IHDR length (4 bytes) + IHDR type (4 bytes) + width (4 bytes) + height (4 bytes)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\r"
        b"IHDR"
        + struct.pack(">II", 800, 600)
        + b"\x08\x02\x00\x00\x00"  # bit depth, color type, compression, filter, interlace
        + b"\x00\x00\x00\x00"  # CRC placeholder
    )
    png_file.write_bytes(png_data)

    width, height = get_image_dimensions(png_file)
    assert width == 800
    assert height == 600


def test_bmp_dimensions(tmp_path: Path):
    bmp_file = tmp_path / "sample.bmp"
    # BM + 12 dummy bytes + DIB header size (40) + width (1024) + height (768)
    bmp_data = (
        b"BM"
        + b"\x00" * 12
        + struct.pack("<Iii", 40, 1024, 768)
    )
    bmp_file.write_bytes(bmp_data)

    width, height = get_image_dimensions(bmp_file)
    assert width == 1024
    assert height == 768


def test_jpeg_dimensions(tmp_path: Path):
    jpeg_file = tmp_path / "sample.jpg"
    # JPEG SOI (\xff\xd8)
    # APP0 marker (\xff\xe0) + length (16) + dummy payload
    # SOF0 marker (\xff\xc0) + length (17) + precision (8) + height (480) + width (640)
    app0_payload = b"\x00" * 14
    app0_chunk = b"\xff\xe0" + struct.pack(">H", len(app0_payload) + 2) + app0_payload

    sof0_payload = struct.pack(">BHH", 8, 480, 640) + b"\x03\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    sof0_chunk = b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload

    jpeg_data = b"\xff\xd8" + app0_chunk + sof0_chunk + b"\xff\xd9"
    jpeg_file.write_bytes(jpeg_data)

    width, height = get_image_dimensions(jpeg_file)
    assert width == 640
    assert height == 480


def test_image_not_found():
    with pytest.raises(FileNotFoundError):
        get_image_dimensions("nonexistent_image.png")


def test_corrupted_or_unsupported_file(tmp_path: Path):
    bad_file = tmp_path / "bad.bin"
    bad_file.write_bytes(b"HELLO_WORLD_NOT_AN_IMAGE")

    with pytest.raises(ValueError, match="Unsupported image format"):
        get_image_dimensions(bad_file)
