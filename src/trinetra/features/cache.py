"""Input fingerprints for content-addressed feature cache validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from trinetra.parsers.models import Dataset


def fingerprint_dataset(dataset: Dataset) -> str:
    """Hash dataset identity, ordered image metadata, and each available source file."""
    digest = hashlib.sha256()
    header = {
        "name": dataset.name,
        "format": str(dataset.format.value if hasattr(dataset.format, "value") else dataset.format),
        "images": len(dataset.images),
    }
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for image in dataset.images:
        path = Path(image.file_path) if image.file_path is not None else None
        identity = {
            "image_id_type": "int" if isinstance(image.image_id, int) else "str",
            "image_id": image.image_id,
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
            "file_path": str(path) if path is not None else None,
        }
        digest.update(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if path is None:
            digest.update(b"|no-path|")
        else:
            try:
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError:
                # Missing/unreadable assets remain fingerprinted and get structured
                # failures from the extractor; they are not silently omitted.
                digest.update(b"|unreadable-or-missing|")
    return f"sha256:{digest.hexdigest()}"


def compute_cache_key(
    *,
    input_fingerprint: str,
    embedding_space_id: str,
    requested_extractor: str,
    actual_extractor: str,
    fallback_used: bool,
    schema_version: int = 3,
) -> str:
    """Combine the source-content digest and complete embedding-space identity."""
    payload = {
        "schema_version": schema_version,
        "input_fingerprint": input_fingerprint,
        "embedding_space_id": embedding_space_id,
        "requested_extractor": requested_extractor,
        "actual_extractor": actual_extractor,
        "fallback_used": fallback_used,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
