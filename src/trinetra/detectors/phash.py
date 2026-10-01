"""Deterministic perceptual-hash duplicate candidates from Layer 1A images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image, ImageOps
except ImportError:  # pragma: no cover - depends on the installation
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]

from trinetra.parsers.models import Dataset, DatasetImage

PHASH_SCHEMA_VERSION = 1
PHASH_DETECTOR_NAME = "phash_duplicate_detection"
PHASH_DETECTOR_VERSION = "1.0"
DEFAULT_PHASH_THRESHOLD = 8
PHASH_SIZE = 8
_DCT_INPUT_SIZE = 32


class PerceptualHashError(ValueError):
    """Invalid Layer 1G input or configuration."""


class ImageBackendUnavailableError(RuntimeError):
    """Pillow is required for pHash image decoding but is unavailable."""


@dataclass(frozen=True)
class PHashFailure:
    image_id: int | str
    file_path: Path | None
    failure_code: str
    reason: str


@dataclass(frozen=True)
class PHashImage:
    image_id: int | str
    file_path: Path
    hash_value: str


@dataclass(frozen=True)
class PHashPair:
    image_id_a: int | str
    image_id_b: int | str
    file_path_a: Path
    file_path_b: Path
    hash_a: str
    hash_b: str
    hamming_distance: int
    configured_threshold: int
    detector_name: str = PHASH_DETECTOR_NAME
    detector_version: str = PHASH_DETECTOR_VERSION


@dataclass(frozen=True)
class PHashClusterMember:
    image_id: int | str
    file_path: Path
    hash_value: str


@dataclass(frozen=True)
class PHashCluster:
    cluster_id: int
    members: tuple[PHashClusterMember, ...]
    pairs: tuple[PHashPair, ...]


@dataclass(frozen=True)
class PHashResult:
    schema_version: int
    detector_name: str
    detector_version: str
    threshold: int
    hash_algorithm: str
    hash_size: int
    total_images: int
    successful_hashes: int
    failed_hashes: int
    candidate_pair_count: int
    clusters: tuple[PHashCluster, ...]
    failures: tuple[PHashFailure, ...]
    hashes: tuple[PHashImage, ...]
    pairs: tuple[PHashPair, ...]


def perceptual_hash(image: "Image.Image") -> int:
    """Compute 64-bit DCT pHash with EXIF orientation and grayscale handling.

    Pixels are converted to grayscale, resized to 32x32 with Pillow bicubic
    interpolation, transformed with an orthonormal 2-D DCT, and the upper-left
    8x8 coefficients are thresholded against their median excluding the DC term.
    """
    if Image is None or ImageOps is None:
        raise ImageBackendUnavailableError("Pillow is required to calculate pHash")
    normalized = ImageOps.exif_transpose(image).convert("L").resize(
        (_DCT_INPUT_SIZE, _DCT_INPUT_SIZE), Image.Resampling.BICUBIC
    )
    pixels = np.asarray(normalized, dtype=np.float64)
    n = _DCT_INPUT_SIZE
    indices = np.arange(n, dtype=np.float64)
    frequencies = indices[:, None]
    positions = indices[None, :]
    transform = np.cos((math.pi / n) * (positions + 0.5) * frequencies)
    transform[0, :] *= math.sqrt(1.0 / n)
    transform[1:, :] *= math.sqrt(2.0 / n)
    coefficients = transform @ pixels @ transform.T
    low = coefficients[:PHASH_SIZE, :PHASH_SIZE]
    median = float(np.median(low.reshape(-1)[1:]))
    bits = low >= median
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int | str, hash_b: int | str) -> int:
    """Return differing bits for two 64-bit integer or 16-digit hex hashes."""
    a = _parse_hash(hash_a)
    b = _parse_hash(hash_b)
    return (a ^ b).bit_count()


class PHashDuplicateDetector:
    """Exact pairwise pHash detector over local Layer 1A image paths."""

    def __init__(self, threshold: int = DEFAULT_PHASH_THRESHOLD) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise PerceptualHashError("threshold must be an integer")
        if not 0 <= threshold <= PHASH_SIZE * PHASH_SIZE:
            raise PerceptualHashError("threshold must be between 0 and 64 for 64-bit pHash")
        self.threshold = threshold

    def detect(self, dataset: Dataset | Iterable[DatasetImage]) -> PHashResult:
        images = dataset.images if isinstance(dataset, Dataset) else list(dataset)
        ids: set[tuple[str, str]] = set()
        ordered: list[DatasetImage] = []
        for image in images:
            key = _id_order(image.image_id)
            if key in ids:
                raise PerceptualHashError(f"Duplicate image_id: {image.image_id!r}")
            ids.add(key)
            ordered.append(image)
        ordered.sort(key=lambda item: _id_order(item.image_id))

        if Image is None or ImageOps is None:
            raise ImageBackendUnavailableError("Pillow is required to load images for pHash")

        hashes: list[PHashImage] = []
        failures: list[PHashFailure] = []
        for image in ordered:
            path = Path(image.file_path) if image.file_path is not None else None
            if path is None:
                failures.append(PHashFailure(image.image_id, None, "missing_path", "No local file_path is available"))
                continue
            try:
                with Image.open(path) as opened:
                    hash_value = perceptual_hash(opened)
                hashes.append(PHashImage(image.image_id, path, f"{hash_value:016x}"))
            except FileNotFoundError:
                failures.append(PHashFailure(image.image_id, path, "missing_image", f"Image file was not found: {path}"))
            except (OSError, ValueError) as exc:
                failures.append(PHashFailure(image.image_id, path, "image_decode_error", f"Could not decode image: {exc}"))
            except Exception as exc:  # keep unexpected hash errors visible per input
                failures.append(PHashFailure(image.image_id, path, "hash_calculation_error", f"Could not calculate pHash: {exc}"))

        pairs: list[PHashPair] = []
        for index, first in enumerate(hashes):
            for second in hashes[index + 1 :]:
                distance = hamming_distance(first.hash_value, second.hash_value)
                if distance <= self.threshold:
                    pairs.append(PHashPair(
                        first.image_id, second.image_id, first.file_path, second.file_path,
                        first.hash_value, second.hash_value, distance, self.threshold,
                    ))
        pairs.sort(key=lambda pair: (_id_order(pair.image_id_a), _id_order(pair.image_id_b)))
        clusters = _build_clusters(hashes, pairs)
        return PHashResult(
            schema_version=PHASH_SCHEMA_VERSION,
            detector_name=PHASH_DETECTOR_NAME,
            detector_version=PHASH_DETECTOR_VERSION,
            threshold=self.threshold,
            hash_algorithm="dct-phash-64",
            hash_size=PHASH_SIZE,
            total_images=len(ordered),
            successful_hashes=len(hashes),
            failed_hashes=len(failures),
            candidate_pair_count=len(pairs),
            clusters=clusters,
            failures=tuple(failures),
            hashes=tuple(hashes),
            pairs=tuple(pairs),
        )


def _parse_hash(value: int | str) -> int:
    if isinstance(value, bool):
        raise PerceptualHashError("hash must be a 64-bit integer or 16-digit hex string")
    if isinstance(value, str):
        try:
            value = int(value, 16)
        except ValueError as exc:
            raise PerceptualHashError("hash string must be hexadecimal") from exc
    if not isinstance(value, int) or not 0 <= value < (1 << 64):
        raise PerceptualHashError("hash must be an unsigned 64-bit value")
    return value


def _id_order(image_id: int | str) -> tuple[str, str]:
    if isinstance(image_id, bool) or not isinstance(image_id, (int, str)):
        raise PerceptualHashError(f"Unsupported image_id: {image_id!r}")
    return ("int" if isinstance(image_id, int) else "str", str(image_id))


def _build_clusters(hashes: list[PHashImage], pairs: list[PHashPair]) -> tuple[PHashCluster, ...]:
    parent = { _id_order(item.image_id): _id_order(item.image_id) for item in hashes }

    def find(item: tuple[str, str]) -> tuple[str, str]:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for pair in pairs:
        a, b = find(_id_order(pair.image_id_a)), find(_id_order(pair.image_id_b))
        if a != b:
            parent[max(a, b)] = min(a, b)
    components: dict[tuple[str, str], list[PHashImage]] = {}
    for item in hashes:
        root = find(_id_order(item.image_id))
        components.setdefault(root, []).append(item)
    output: list[PHashCluster] = []
    for members in components.values():
        if len(members) < 2:
            continue
        member_keys = {_id_order(item.image_id) for item in members}
        component_pairs = tuple(pair for pair in pairs if _id_order(pair.image_id_a) in member_keys and _id_order(pair.image_id_b) in member_keys)
        output.append(PHashCluster(
            cluster_id=0,
            members=tuple(PHashClusterMember(item.image_id, item.file_path, item.hash_value) for item in sorted(members, key=lambda item: _id_order(item.image_id))),
            pairs=component_pairs,
        ))
    output.sort(key=lambda group: _id_order(group.members[0].image_id))
    return tuple(PHashCluster(index + 1, group.members, group.pairs) for index, group in enumerate(output))
