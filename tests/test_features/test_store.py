"""Tests for embedding save/load persistence."""

from pathlib import Path

import numpy as np
import pytest
import json

from trinetra.features.exceptions import EmbeddingStoreError, TrinetraFeatureError
from trinetra.features.models import FeatureBatch, ImageEmbedding, ImageFailure, make_embedding_space_id
from trinetra.features.store import EmbeddingStore, SCHEMA_VERSION


def _sample_batch(tmp_path: Path) -> FeatureBatch:
    return FeatureBatch(
        dataset_name="unit set",
        backend="dummy",
        model_id="dummy-meanpool-v1",
        embedding_dim=4,
        device="cpu",
        embeddings=[
            ImageEmbedding(
                image_id=101,
                file_name="a.png",
                vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                file_path=tmp_path / "a.png",
            ),
            ImageEmbedding(
                image_id="frame_002",
                file_name="b.png",
                vector=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                warnings=["note"],
            ),
        ],
        warnings=["skipped missing.png"],
        preprocessing={"resize": [32, 32], "mode": "RGB"},
        normalization={"method": "l2"},
        weights_identity="sha256:abc",
        failures=[ImageFailure(202, tmp_path / "bad.png", "image_decode_error", "bad raster")],
        input_fingerprint="sha256:input-v1",
    )


def test_embedding_store_roundtrip(tmp_path: Path):
    store = EmbeddingStore()
    original = _sample_batch(tmp_path)
    json_path = store.save(original, tmp_path / "embeddings")

    assert json_path.is_file()
    assert json_path.with_suffix(".npz").is_file()

    loaded = store.load(json_path)
    assert loaded.dataset_name == original.dataset_name
    assert loaded.backend == "dummy"
    assert loaded.model_id == "dummy-meanpool-v1"
    assert loaded.embedding_dim == 4
    assert loaded.device == "cpu"
    assert loaded.warnings == ["skipped missing.png"]
    assert loaded.failures == original.failures
    assert loaded.preprocessing == original.preprocessing
    assert loaded.normalization == original.normalization
    assert loaded.embedding_space_id == original.embedding_space_id
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.requested_extractor == loaded.actual_extractor == "dummy"
    assert not loaded.fallback_used
    assert loaded.input_fingerprint == original.input_fingerprint
    assert loaded.cache_key == original.cache_key
    assert loaded.embedding_count == 2
    assert loaded.embeddings[0].image_id == 101
    assert loaded.embeddings[1].image_id == "frame_002"
    assert loaded.embeddings[0].file_path == tmp_path / "a.png"
    assert loaded.embeddings[1].file_path is None
    assert loaded.embeddings[1].warnings == ["note"]
    np.testing.assert_array_equal(loaded.as_matrix(), original.as_matrix())

    from_npz = store.load(json_path.with_suffix(".npz"))
    assert from_npz.embedding_count == 2


def test_embedding_store_empty_batch(tmp_path: Path):
    store = EmbeddingStore()
    empty = FeatureBatch(
        dataset_name="none",
        backend="dummy",
        model_id="dummy-meanpool-v1",
        embedding_dim=8,
        device="cpu",
        preprocessing={"method": "unit-test"},
        normalization={"method": "none"},
        warnings=["nothing readable"],
        input_fingerprint="sha256:empty-v1",
    )
    json_path = store.save(empty, tmp_path)
    loaded = store.load(json_path)
    assert loaded.embedding_count == 0
    assert loaded.as_matrix().shape == (0, 8)
    assert loaded.warnings == ["nothing readable"]


def test_embedding_space_id_changes_with_each_compatibility_input():
    common = dict(model_id="m", weights_identity="w", preprocessing={"resize": 4},
                  embedding_dim=3, normalization={"method": "l2"})
    base = make_embedding_space_id(**common)
    for key, value in (("model_id", "m2"), ("weights_identity", "w2"),
                       ("preprocessing", {"resize": 8}), ("embedding_dim", 4),
                       ("normalization", {"method": "none"})):
        changed = dict(common)
        changed[key] = value
        assert make_embedding_space_id(**changed) != base


def test_duplicate_image_id_is_rejected():
    vector = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(TrinetraFeatureError, match="Duplicate image_id"):
        FeatureBatch("d", "b", "m", 2, "cpu", embeddings=[
            ImageEmbedding(1, "a", vector), ImageEmbedding(1, "b", vector)
        ], preprocessing={"method": "unit-test"}, normalization={"method": "none"})


def test_missing_schema_version_and_mismatched_vector_count_rejected(tmp_path: Path):
    store = EmbeddingStore()
    path = store.save(_sample_batch(tmp_path), tmp_path / "store")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    del metadata["schema_version"]
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(EmbeddingStoreError, match="missing schema_version"):
        store.load(path)

    metadata["schema_version"] = SCHEMA_VERSION
    metadata["embeddings"].pop()
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(EmbeddingStoreError, match="Vector matrix shape"):
        store.load(path)


def test_failed_second_replace_restores_existing_pair(tmp_path: Path, monkeypatch):
    import trinetra.features.store as store_module

    store = EmbeddingStore()
    original = _sample_batch(tmp_path)
    json_path = store.save(original, tmp_path / "atomic")
    old_json, old_npz = json_path.read_bytes(), json_path.with_suffix(".npz").read_bytes()
    replacement = FeatureBatch("other", "dummy", "m2", 2, "cpu",
                               preprocessing={"method": "unit-test"}, normalization={"method": "none"},
                               input_fingerprint="sha256:replacement-v1",
                               embeddings=[ImageEmbedding("x", "x.png", np.array([1, 0], dtype=np.float32))])
    real_replace = store_module.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        real_replace(source, target)

    monkeypatch.setattr(store_module.os, "replace", fail_second)
    with pytest.raises(EmbeddingStoreError, match="atomically write"):
        store.save(replacement, tmp_path / "atomic", stem=json_path.stem)
    assert json_path.read_bytes() == old_json
    assert json_path.with_suffix(".npz").read_bytes() == old_npz


def test_embedding_store_rejects_stale_cache_key(tmp_path: Path):
    store = EmbeddingStore()
    path = store.save(_sample_batch(tmp_path), tmp_path / "stale")
    with pytest.raises(EmbeddingStoreError, match="cache identity mismatch"):
        store.load(path, expected_cache_key="sha256:changed-input-or-model")


def test_embedding_store_round_trips_fallback_provenance(tmp_path: Path):
    store = EmbeddingStore()
    batch = FeatureBatch(
        dataset_name="fallback", backend="resnet50", model_id="resnet50-imagenet-avgpool-v1",
        embedding_dim=2048, device="cpu", preprocessing={"resize": 256}, normalization={"method": "l2"},
        weights_identity="sha256:resnet", requested_extractor="dinov2", actual_extractor="resnet50",
        fallback_used=True, fallback_reason="DINO weights unavailable", input_fingerprint="sha256:dataset",
        embeddings=[ImageEmbedding("i", "i.png", np.ones(2048, dtype=np.float32))],
    )
    loaded = store.load(store.save(batch, tmp_path))
    assert loaded.requested_extractor == "dinov2"
    assert loaded.actual_extractor == "resnet50"
    assert loaded.fallback_used
    assert loaded.fallback_reason == "DINO weights unavailable"


def test_embedding_store_missing_files(tmp_path: Path):
    store = EmbeddingStore()
    with pytest.raises(EmbeddingStoreError, match="metadata file not found"):
        store.load(tmp_path / "missing.json")

    json_path = tmp_path / "orphan.json"
    json_path.write_text("{}", encoding="utf-8")
    with pytest.raises(EmbeddingStoreError, match="matrix file not found"):
        store.load(json_path)
