from pathlib import Path

import numpy as np
import pytest

from trinetra.detectors import (
    DEFAULT_SIMILARITY_THRESHOLD,
    EmbeddingSpaceMismatchError,
    InvalidFeatureBatchError,
    NearDuplicateDetector,
    NearDuplicateResult,
    cosine_similarity,
)
from trinetra.features.models import FeatureBatch, ImageEmbedding


def _batch(vectors, ids=None):
    ids = ids or list(range(len(vectors)))
    embeddings = [
        ImageEmbedding(image_id, f"{image_id}.png", np.asarray(vector, dtype=np.float32), Path(f"images/{image_id}.png"))
        for image_id, vector in zip(ids, vectors, strict=True)
    ]
    return FeatureBatch(
        dataset_name="synthetic",
        backend="mock",
        model_id="mock-v1",
        weights_identity="sha256:mockweights",
        embedding_dim=len(vectors[0]) if vectors else 2,
        device="cpu",
        embeddings=embeddings,
        preprocessing={"method": "synthetic"},
        normalization={"method": "l2"},
    )


def test_identical_vectors_similarity_one_and_result_has_provenance():
    batch = _batch([[1, 0], [1, 0]])
    result = NearDuplicateDetector().detect(batch)

    assert cosine_similarity(np.array([2.0, 0.0]), np.array([3.0, 0.0])) == 1.0
    assert isinstance(result, NearDuplicateResult)
    assert result.schema_version == 1
    assert result.threshold == DEFAULT_SIMILARITY_THRESHOLD
    assert result.embedding_space_id == batch.embedding_space_id
    assert result.total_images == 2
    assert result.candidate_pair_count == 1
    pair = result.pairs[0]
    assert (pair.image_a_id, pair.image_b_id) == (0, 1)
    assert pair.image_a_path == Path("images/0.png")
    assert pair.image_b_path == Path("images/1.png")
    assert pair.similarity == 1.0
    assert pair.threshold == result.threshold
    assert pair.embedding_space_id == batch.embedding_space_id


def test_high_similarity_detected_and_below_threshold_not_detected():
    near = _batch([[1, 0], [0.99, 0.1]])
    far = _batch([[1, 0], [0, 1]])
    assert NearDuplicateDetector(0.95).detect(near).candidate_pair_count == 1
    assert NearDuplicateDetector(0.95).detect(far).candidate_pair_count == 0


def test_threshold_is_configurable_and_validated():
    result = NearDuplicateDetector(0.7).detect(_batch([[1, 0], [0.8, 0.6]]))
    assert result.threshold == 0.7
    assert result.candidate_pair_count == 1
    with pytest.raises(ValueError, match="threshold"):
        NearDuplicateDetector(1.1)


def test_self_comparison_and_pair_duplicates_are_excluded():
    result = NearDuplicateDetector(0.9).detect(_batch([[1, 0], [1, 0], [0, 1]]))
    assert [(pair.image_a_id, pair.image_b_id) for pair in result.pairs] == [(0, 1)]
    assert result.candidate_pair_count == 1


def test_pair_and_cluster_ordering_is_deterministic_for_input_order():
    vectors = [[1, 0], [0.99, 0.1], [0.98, 0.2]]
    first = NearDuplicateDetector(0.95).detect(_batch(vectors, ["c", "a", "b"]))
    second = NearDuplicateDetector(0.95).detect(_batch(vectors, ["b", "c", "a"]))
    pair_ids = lambda result: [(pair.image_a_id, pair.image_b_id) for pair in result.pairs]
    assert pair_ids(first) == pair_ids(second) == [("a", "b"), ("a", "c"), ("b", "c")]
    assert [[member.image_id for member in cluster.members] for cluster in first.clusters] == [
        [member.image_id for member in cluster.members] for cluster in second.clusters
    ]


def test_mixed_embedding_space_record_is_rejected():
    batch = _batch([[1, 0], [0.99, 0.1]])
    batch.embeddings[1].embedding_space_id = "sha256:another-space"
    with pytest.raises(EmbeddingSpaceMismatchError, match="belongs to embedding space"):
        NearDuplicateDetector().detect(batch)


def test_inconsistent_batch_embedding_space_is_rejected():
    batch = _batch([[1, 0]])
    batch.embedding_space_id = "sha256:wrong"
    with pytest.raises(EmbeddingSpaceMismatchError, match="does not match"):
        NearDuplicateDetector().detect(batch)


def test_dimension_mismatch_is_rejected():
    batch = _batch([[1, 0], [0, 1]])
    batch.embeddings[1].vector = np.array([0, 1, 2], dtype=np.float32)
    with pytest.raises(InvalidFeatureBatchError, match="dimension"):
        NearDuplicateDetector().detect(batch)


def test_zero_vectors_are_safe_and_do_not_create_default_candidate():
    assert cosine_similarity(np.zeros(2), np.array([1, 0])) == 0.0
    result = NearDuplicateDetector().detect(_batch([[0, 0], [1, 0]]))
    assert result.candidate_pair_count == 0


@pytest.mark.parametrize("invalid", [np.array([np.nan, 0]), np.array([np.inf, 1])])
def test_nonfinite_embeddings_rejected(invalid):
    batch = _batch([[1, 0], [0, 1]])
    batch.embeddings[0].vector = invalid
    with pytest.raises(InvalidFeatureBatchError, match="finite"):
        NearDuplicateDetector().detect(batch)


def test_cosine_rejects_dimension_mismatch_and_nonfinite_values():
    with pytest.raises(InvalidFeatureBatchError, match="dimension"):
        cosine_similarity(np.ones(2), np.ones(3))
    with pytest.raises(InvalidFeatureBatchError, match="finite"):
        cosine_similarity(np.array([np.nan]), np.array([1.0]))


def test_connected_relationships_form_clusters_and_disconnected_groups_stay_separate():
    angles = [0, 20, 40, 180, 100, 120]
    vectors = [[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))] for angle in angles]
    result = NearDuplicateDetector(0.9).detect(_batch(vectors, ["a", "b", "c", "x", "y", "z"]))

    assert result.candidate_pair_count == 3
    assert result.duplicate_cluster_count == 2
    assert [[member.image_id for member in cluster.members] for cluster in result.clusters] == [
        ["a", "b", "c"], ["y", "z"]
    ]
    assert all(pair.embedding_space_id == result.embedding_space_id for pair in result.pairs)
    assert [[(pair.image_a_id, pair.image_b_id) for pair in cluster.pairs] for cluster in result.clusters] == [
        [("a", "b"), ("b", "c")], [("y", "z")]
    ]


def test_empty_and_single_image_batches_have_empty_results():
    empty = NearDuplicateDetector().detect(_batch([]))
    single = NearDuplicateDetector().detect(_batch([[1, 0]]))
    for result in (empty, single):
        assert result.total_images in {0, 1}
        assert result.pairs == ()
        assert result.clusters == ()
        assert result.candidate_pair_count == 0
        assert result.duplicate_cluster_count == 0
