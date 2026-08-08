import pytest
import pandas as pd
import numpy as np
from src.embedding_generator import (
    EmbeddingGenerator,
    get_embedding_model,
    reset_model_singleton,
    compute_dataset_hash,
    clear_embedding_cache,
    get_embedding_cache,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Clear cache and reset singletons before each test."""
    clear_embedding_cache()
    yield
    clear_embedding_cache()


@pytest.fixture
def generator():
    return EmbeddingGenerator()


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "Need dark theme"},
        {"ticket_id": 2, "topic": "CSV Export", "message": "Download CSV data"}
    ])


def test_model_singleton_reuse():
    """Verify that get_embedding_model returns the exact same singleton instance."""
    reset_model_singleton()
    model1 = get_embedding_model()
    model2 = get_embedding_model()
    assert model1 is model2


def test_compute_dataset_hash_consistency(sample_df):
    """Verify compute_dataset_hash returns deterministic MD5 hashes."""
    hash1 = compute_dataset_hash(sample_df)
    hash2 = compute_dataset_hash(sample_df)
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 32  # MD5 hex digest length


def test_encode_single_text(generator):
    """Verify encoding a single string returns a normalized 2D vector array."""
    emb = generator.encode_texts("Dark mode proposal")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 384)
    # Check L2 normalization
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, abs=1e-3) == 1.0


def test_encode_multiple_texts(generator):
    """Verify encoding multiple strings returns a 2D matrix of shape (N, 384)."""
    texts = ["Dark mode", "CSV export", "App crash"]
    emb = generator.encode_texts(texts)
    assert emb.shape == (3, 384)


def test_encode_empty_text_raises_error(generator):
    """Verify empty or whitespace strings raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        generator.encode_texts("")

    with pytest.raises(ValueError, match="cannot be empty"):
        generator.encode_texts("   ")

    with pytest.raises(ValueError, match="cannot be empty"):
        generator.encode_texts([])


def test_encode_invalid_type_raises_error(generator):
    """Verify non-string/list inputs raise TypeError."""
    with pytest.raises(TypeError, match="must be a string or a list of strings"):
        generator.encode_texts(12345)


def test_encode_dataframe_caching(generator, sample_df):
    """Verify encode_dataframe computes embeddings and reuses cache on second call."""
    # First call - cache miss
    dataset_hash = compute_dataset_hash(sample_df)
    cache = get_embedding_cache()
    assert dataset_hash not in cache

    emb1 = generator.encode_dataframe(sample_df)
    assert emb1.shape == (2, 384)
    assert dataset_hash in cache

    # Second call - cache hit (returns exact same numpy array reference)
    emb2 = generator.encode_dataframe(sample_df)
    assert emb2 is emb1


def test_encode_dataframe_missing_columns(generator):
    """Verify DataFrame missing required topic/message columns raises ValueError."""
    invalid_df = pd.DataFrame({"id": [1], "content": ["Hello"]})
    with pytest.raises(ValueError, match="DataFrame missing required columns"):
        generator.encode_dataframe(invalid_df)
