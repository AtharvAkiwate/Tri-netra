"""Minimal pure-Python image header dimension reader for TRI-NETRA.

Extracts width and height for JPEG, PNG, and BMP image files without external dependencies.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Union


def get_image_dimensions(file_path: Union[str, Path]) -> tuple[int, int]:
    """Extract width and height in pixels from an image file header.

    Supported formats: JPEG, PNG, BMP.
    Does not decode full image rasters; only reads the minimal header bytes.

    Args:
        file_path: Path to the image file on disk.

    Returns:
        Tuple of (width, height) in pixels.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a supported format or has a corrupted header.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with path.open("rb") as f:
        header = f.read(32)
        if len(header) < 16:
            raise ValueError(f"File too small to be a valid image: {path}")

        # 1. PNG check
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            # PNG signature (8 bytes) + IHDR length (4 bytes) + chunk type 'IHDR' (4 bytes)
            # Offset 16 to 24 contains width (4 bytes >I) and height (4 bytes >I)
            if len(header) < 24:
                raise ValueError(f"Corrupted PNG header in {path}")
            width, height = struct.unpack(">II", header[16:24])
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid PNG dimensions ({width}x{height}) in {path}")
            return (width, height)

        # 2. BMP check
        if header.startswith(b"BM"):
            if len(header) < 26:
                raise ValueError(f"Corrupted BMP header in {path}")
            dib_header_size = struct.unpack("<I", header[14:18])[0]
            if dib_header_size >= 40:
                width, height = struct.unpack("<ii", header[18:26])
                height = abs(height)
            elif dib_header_size == 12:
                width, height = struct.unpack("<HH", header[18:22])
            else:
                raise ValueError(f"Unsupported BMP DIB header size ({dib_header_size}) in {path}")

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid BMP dimensions ({width}x{height}) in {path}")
            return (int(width), int(height))

        # 3. JPEG check
        if header.startswith(b"\xff\xd8"):
            f.seek(2)
            # SOF marker codes (Baseline, Extended, Progressive, Lossless, Differential)
            sof_markers = {
                0xC0, 0xC1, 0xC2, 0xC3,
                0xC5, 0xC6, 0xC7,
                0xC9, 0xCA, 0xCB,
                0xCD, 0xCE, 0xCF,
            }
            # Standalone markers with no length payload (SOI, EOI, RST0..RST7)
            standalone_markers = {0xD8, 0xD9} | set(range(0xD0, 0xD8))

            while True:
                # Seek next marker (starting with 0xFF)
                byte = f.read(1)
                if not byte:
                    break
                if byte[0] != 0xFF:
                    continue

                # Consume any consecutive 0xFF padding bytes
                marker_byte = f.read(1)
                while marker_byte and marker_byte[0] == 0xFF:
                    marker_byte = f.read(1)

                if not marker_byte:
                    break

                marker_code = marker_byte[0]

                if marker_code in standalone_markers:
                    continue

                length_bytes = f.read(2)
                if len(length_bytes) < 2:
                    break
                length = struct.unpack(">H", length_bytes)[0]

                if marker_code in sof_markers:
                    # Payload: precision (1 byte), height (2 bytes), width (2 bytes)
                    sof_data = f.read(5)
                    if len(sof_data) < 5:
                        raise ValueError(f"Corrupted JPEG SOF chunk in {path}")
                    _, height, width = struct.unpack(">BHH", sof_data)
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Invalid JPEG dimensions ({width}x{height}) in {path}")
                    return (int(width), int(height))

                # Skip rest of this marker payload
                skip_bytes = length - 2
                if skip_bytes > 0:
                    f.seek(skip_bytes, 1)

            raise ValueError(f"Could not find valid frame header (SOF) in JPEG {path}")

    raise ValueError(f"Unsupported image format or signature for: {path}")
