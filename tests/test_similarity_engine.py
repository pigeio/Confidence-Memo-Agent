import pytest
import numpy as np
from src.similarity_engine import SimilarityEngine


def test_orthogonal_vectors():
    """Verify that orthogonal vectors produce a 0.0 similarity score."""
    ticket_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    query_embedding = np.array([0.0, 1.0], dtype=np.float32)

    scores = SimilarityEngine.compute_cosine_similarity(ticket_embeddings, query_embedding)
    assert len(scores) == 1
    assert pytest.approx(scores[0], abs=1e-5) == 0.0


def test_identical_vectors():
    """Verify that identical normalized vectors produce a 1.0 similarity score."""
    ticket_embeddings = np.array([[0.6, 0.8]], dtype=np.float32)
    query_embedding = np.array([0.6, 0.8], dtype=np.float32)

    scores = SimilarityEngine.compute_cosine_similarity(ticket_embeddings, query_embedding)
    assert len(scores) == 1
    assert pytest.approx(scores[0], abs=1e-5) == 1.0


def test_opposite_vectors():
    """Verify that opposite normalized vectors produce a -1.0 similarity score."""
    ticket_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    query_embedding = np.array([-1.0, 0.0], dtype=np.float32)

    scores = SimilarityEngine.compute_cosine_similarity(ticket_embeddings, query_embedding)
    assert pytest.approx(scores[0], abs=1e-5) == -1.0


def test_multiple_tickets_similarity():
    """Verify similarity computation across multiple candidate vectors."""
    ticket_embeddings = np.array([
        [1.0, 0.0],   # Ticket 0: identical direction -> score 1.0
        [0.0, 1.0],   # Ticket 1: orthogonal -> score 0.0
        [0.7071, 0.7071] # Ticket 2: ~45 deg -> score ~0.7071
    ], dtype=np.float32)
    query_embedding = np.array([1.0, 0.0], dtype=np.float32)

    scores = SimilarityEngine.compute_cosine_similarity(ticket_embeddings, query_embedding)
    assert len(scores) == 3
    assert pytest.approx(scores[0], abs=1e-4) == 1.0
    assert pytest.approx(scores[1], abs=1e-4) == 0.0
    assert pytest.approx(scores[2], abs=1e-4) == 0.7071


def test_rank_and_filter_sorting_and_threshold():
    """Verify rank_and_filter ranks candidates descending and filters out scores below threshold."""
    scores = np.array([0.1, 0.85, 0.4, 0.95, 0.05], dtype=np.float32)
    ranked = SimilarityEngine.rank_and_filter(scores, top_k=5, threshold=0.3)

    # Expected order: index 3 (0.95), index 1 (0.85), index 2 (0.4)
    # Indices 0 (0.1) and 4 (0.05) are below 0.3 threshold
    assert len(ranked) == 3
    assert ranked[0] == (3, pytest.approx(0.95, abs=1e-4))
    assert ranked[1] == (1, pytest.approx(0.85, abs=1e-4))
    assert ranked[2] == (2, pytest.approx(0.4, abs=1e-4))


def test_rank_and_filter_top_k_limiting():
    """Verify rank_and_filter truncates results to top_k limit."""
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    ranked = SimilarityEngine.rank_and_filter(scores, top_k=2, threshold=0.5)

    assert len(ranked) == 2
    assert ranked[0][0] == 0
    assert ranked[1][0] == 1


def test_empty_embeddings_handling():
    """Verify empty input arrays return empty results gracefully."""
    empty_tickets = np.empty((0, 384), dtype=np.float32)
    query = np.ones(384, dtype=np.float32)

    scores = SimilarityEngine.compute_cosine_similarity(empty_tickets, query)
    assert len(scores) == 0

    ranked = SimilarityEngine.rank_and_filter(scores, top_k=5, threshold=0.2)
    assert ranked == []


def test_dimension_mismatch_raises_error():
    """Verify dimension mismatch between ticket embeddings and query raises ValueError."""
    ticket_embeddings = np.ones((5, 384), dtype=np.float32)
    query_512 = np.ones(512, dtype=np.float32)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        SimilarityEngine.compute_cosine_similarity(ticket_embeddings, query_512)


def test_invalid_types_raise_error():
    """Verify non-ndarray types raise TypeError."""
    with pytest.raises(TypeError, match="must be numpy ndarrays"):
        SimilarityEngine.compute_cosine_similarity([[1, 0]], np.array([1, 0]))

    with pytest.raises(TypeError, match="must be a numpy ndarray"):
        SimilarityEngine.rank_and_filter([0.8, 0.5])
