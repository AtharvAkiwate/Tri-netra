"""Tests for feature-extraction data models."""

import numpy as np
import pytest

from trinetra.features.exceptions import TrinetraFeatureError
from trinetra.features.models import FeatureBatch, ImageEmbedding


def test_image_embedding_canonicalizes_float32_vector():
    item = ImageEmbedding(
        image_id=7,
        file_name="a.jpg",
        vector=np.array([0.3, 0.4], dtype=np.float64),
    )
    assert item.vector.dtype == np.float32
    assert item.vector.shape == (2,)
    np.testing.assert_allclose(item.vector, [0.3, 0.4], rtol=1e-6)


def test_image_embedding_rejects_non_finite_or_wrong_rank():
    with pytest.raises(TrinetraFeatureError, match="NaN or infinite"):
        ImageEmbedding(image_id="x", file_name="a.jpg", vector=np.array([1.0, np.nan]))

    with pytest.raises(TrinetraFeatureError, match="1-D"):
        ImageEmbedding(image_id="x", file_name="a.jpg", vector=np.zeros((2, 2), dtype=np.float32))


def test_feature_batch_matrix_and_dim_validation():
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    batch = FeatureBatch(
        dataset_name="demo",
        backend="dummy",
        model_id="dummy-meanpool-v1",
        embedding_dim=3,
        device="cpu",
        preprocessing={"method": "unit-test"},
        normalization={"method": "none"},
        embeddings=[
            ImageEmbedding(image_id=1, file_name="a.png", vector=vec_a),
            ImageEmbedding(image_id="frame_2", file_name="b.png", vector=vec_b),
        ],
        warnings=["skipped one image"],
    )
    assert batch.embedding_count == 2
    matrix = batch.as_matrix()
    assert matrix.shape == (2, 3)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], vec_a)

    with pytest.raises(TrinetraFeatureError, match="does not match embedding_dim"):
        FeatureBatch(
            dataset_name="demo",
            backend="dummy",
            model_id="x",
            embedding_dim=3,
            device="cpu",
            preprocessing={"method": "unit-test"},
            normalization={"method": "none"},
            embeddings=[ImageEmbedding(image_id=1, file_name="a.png", vector=np.array([1.0], dtype=np.float32))],
        )


def test_empty_feature_batch_matrix_uses_declared_dim():
    batch = FeatureBatch(
        dataset_name="empty",
        backend="dummy",
        model_id="dummy-meanpool-v1",
        embedding_dim=8,
        device="cpu",
        preprocessing={"method": "unit-test"},
        normalization={"method": "none"},
    )
    matrix = batch.as_matrix()
    assert matrix.shape == (0, 8)
    assert matrix.dtype == np.float32
