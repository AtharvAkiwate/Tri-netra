from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from trinetra.detectors.phash import (
    PHashDuplicateDetector,
    PerceptualHashError,
    hamming_distance,
    perceptual_hash,
)
from trinetra.parsers.models import DatasetImage


def _image(image_id, path):
    return DatasetImage(image_id, Path(path).name, 64, 64, file_path=Path(path))


def _save(tmp_path, name, image):
    path = tmp_path / name
    image.save(path)
    return path


def test_identical_images_have_zero_distance(tmp_path):
    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((8, 8, 30, 50), fill="black")
    first = _save(tmp_path, "a.png", image)
    second = _save(tmp_path, "b.png", image.copy())
    result = PHashDuplicateDetector().detect([_image("a", first), _image("b", second)])
    assert result.pairs[0].hamming_distance == 0
    assert result.pairs[0].image_id_a == "a"
    assert result.pairs[0].image_id_b == "b"


def test_similar_and_different_images_and_threshold(tmp_path):
    base = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(base).rectangle((8, 8, 30, 50), fill="black")
    similar = base.copy()
    ImageDraw.Draw(similar).rectangle((9, 8, 31, 50), fill="black")
    different = Image.new("RGB", (64, 64), "black")
    ImageDraw.Draw(different).ellipse((8, 8, 56, 56), fill="white")
    paths = [_save(tmp_path, f"{i}.png", im) for i, im in enumerate((base, similar, different))]
    default_result = PHashDuplicateDetector().detect([_image(i, path) for i, path in enumerate(paths)])
    assert any({pair.image_id_a, pair.image_id_b} == {0, 1} for pair in default_result.pairs)
    result = PHashDuplicateDetector(threshold=0).detect([_image(i, path) for i, path in enumerate(paths)])
    assert any(pair.hamming_distance == 0 for pair in result.pairs)
    assert len(result.pairs) == 1
    assert hamming_distance(perceptual_hash(base), perceptual_hash(similar)) <= 8


def test_ordering_and_unordered_pair_uniqueness(tmp_path):
    image = Image.new("RGB", (32, 32), "gray")
    paths = [_save(tmp_path, f"{name}.png", image) for name in ("z", "b", "a")]
    result = PHashDuplicateDetector().detect([_image(name, path) for name, path in zip(("z", "b", "a"), paths)])
    pairs = [(pair.image_id_a, pair.image_id_b) for pair in result.pairs]
    assert pairs == [("a", "b"), ("a", "z"), ("b", "z")]
    assert len(pairs) == len(set(pairs))
    assert all(a < b for a, b in pairs)
    assert len(result.clusters) == 1
    assert [item.image_id for item in result.clusters[0].members] == ["a", "b", "z"]


def test_empty_single_missing_and_corrupt_inputs(tmp_path):
    assert PHashDuplicateDetector().detect([]).total_images == 0
    good = _save(tmp_path, "ok.png", Image.new("RGB", (8, 8), "white"))
    missing = tmp_path / "missing.png"
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"not an image")
    result = PHashDuplicateDetector().detect([
        _image("one", good), _image("missing", missing), _image("bad", corrupt)
    ])
    assert result.successful_hashes == 1
    assert result.failed_hashes == 2
    assert {failure.failure_code for failure in result.failures} == {"missing_image", "image_decode_error"}
    single = PHashDuplicateDetector().detect([_image("one", good)])
    assert single.candidate_pair_count == single.failed_hashes == 0


def test_duplicate_ids_and_invalid_threshold_rejected(tmp_path):
    image = _save(tmp_path, "x.png", Image.new("RGB", (4, 4), "white"))
    with pytest.raises(PerceptualHashError, match="Duplicate"):
        PHashDuplicateDetector().detect([_image("x", image), _image("x", image)])
    for value in (-1, 65, 1.5, True):
        with pytest.raises(PerceptualHashError):
            PHashDuplicateDetector(value)


def test_missing_path_is_structured_failure_and_schema_is_versioned():
    result = PHashDuplicateDetector().detect([DatasetImage("x", "x.png", 0, 0)])
    assert result.schema_version == 1
    assert result.detector_name == "phash_duplicate_detection"
    assert result.hash_algorithm == "dct-phash-64"
    assert result.hash_size == 8
    assert result.failures[0].failure_code == "missing_path"
    assert "file_path" in result.failures[0].__dataclass_fields__


def test_hamming_distance_validates_hash_values():
    assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64
    with pytest.raises(PerceptualHashError):
        hamming_distance(-1, 0)
