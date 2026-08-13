import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.evidence_deduplication import EvidenceDeduplicator, _normalize_text


def test_normalize_text():
    assert _normalize_text("  Hello, World!  ") == "hello world"
    assert _normalize_text("UPPER   CASE\t\nTEST") == "upper case test"
    assert _normalize_text(None) == ""
    assert _normalize_text(123) == "123"


def test_deduplicator_initialization():
    dedup = EvidenceDeduplicator(similarity_threshold=0.8)
    assert dedup.similarity_threshold == 0.8

    with pytest.raises(ValueError, match="similarity_threshold must be between"):
        EvidenceDeduplicator(similarity_threshold=1.5)

    with pytest.raises(ValueError, match="similarity_threshold must be between"):
        EvidenceDeduplicator(similarity_threshold=-0.1)


def test_deduplicate_empty_and_single_df():
    dedup = EvidenceDeduplicator()

    # Empty df
    empty_df = pd.DataFrame(columns=["topic", "message"])
    res_df, res_scores, stats = dedup.deduplicate(empty_df)
    assert res_df.empty
    assert res_scores is None
    assert stats["total_input_count"] == 0
    assert stats["unique_count"] == 0
    assert stats["duplicate_count"] == 0
    assert stats["duplicate_rate"] == 0.0

    # Single ticket df
    single_df = pd.DataFrame([{"topic": "Login", "message": "Can't log in"}])
    scores = np.array([0.9], dtype=np.float32)
    res_df, res_scores, stats = dedup.deduplicate(single_df, similarity_scores=scores)
    assert len(res_df) == 1
    assert len(res_scores) == 1
    assert stats["total_input_count"] == 1
    assert stats["unique_count"] == 1
    assert stats["duplicate_count"] == 0


def test_deduplicate_type_and_value_errors():
    dedup = EvidenceDeduplicator()

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        dedup.deduplicate("not a df")

    with pytest.raises(ValueError, match="representative_strategy must be one of"):
        dedup.deduplicate(pd.DataFrame(), representative_strategy="invalid_strat")

    with pytest.raises(TypeError, match="similarity_scores must be a numpy ndarray"):
        dedup.deduplicate(
            pd.DataFrame([{"topic": "A", "message": "B"}]),
            similarity_scores="invalid_scores",
        )

    with pytest.raises(ValueError, match="similarity_scores length .* must match"):
        dedup.deduplicate(
            pd.DataFrame([{"topic": "A", "message": "B"}]),
            similarity_scores=np.array([0.5, 0.8]),
        )

    with pytest.raises(ValueError, match="df is missing required columns"):
        dedup.deduplicate(pd.DataFrame([{"colA": "valA"}, {"colA": "valB"}]))


def test_exact_duplicates_detection():
    # Mock embedding generator so semantic check doesn't need external model calls in quick test
    mock_embedder = MagicMock()
    # 3 tickets: 0 and 1 are exact duplicates (different casing/punct), 2 is distinct
    df = pd.DataFrame(
        [
            {"ticket_id": 101, "topic": "Dark Mode", "message": "Please add dark mode!"},
            {"ticket_id": 102, "topic": "DARK MODE", "message": "please add dark mode"},
            {"ticket_id": 103, "topic": "Export", "message": "We need CSV export feature"},
        ]
    )
    # Return orthogonal embeddings so only exact match triggers
    mock_embedder.encode_texts.return_value = np.array(
        [[1, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
    )

    dedup = EvidenceDeduplicator(similarity_threshold=0.85, embedding_generator=mock_embedder)
    scores = np.array([0.95, 0.90, 0.80], dtype=np.float32)

    res_df, res_scores, stats = dedup.deduplicate(df, similarity_scores=scores)

    assert len(res_df) == 2
    assert stats["total_input_count"] == 3
    assert stats["unique_count"] == 2
    assert stats["duplicate_count"] == 1
    assert stats["exact_duplicates_count"] == 1
    assert stats["semantic_duplicates_count"] == 0
    assert len(res_scores) == 2


def test_semantic_duplicates_detection():
    mock_embedder = MagicMock()
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "Dark Theme", "message": "The white background hurts my eyes, need night mode"},
            {"ticket_id": 2, "topic": "Night Mode Request", "message": "App is too bright, please provide dark mode options"},
            {"ticket_id": 3, "topic": "Billing Error", "message": "Charged twice on invoice 9912"},
        ]
    )

    # Embeddings: 1 and 2 are highly similar (~0.95 cosine sim), 3 is orthogonal
    vec1 = np.array([0.9, 0.1, 0.0], dtype=np.float32)
    vec1 /= np.linalg.norm(vec1)
    vec2 = np.array([0.88, 0.12, 0.0], dtype=np.float32)
    vec2 /= np.linalg.norm(vec2)
    vec3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    mock_embedder.encode_texts.return_value = np.vstack([vec1, vec2, vec3])

    dedup = EvidenceDeduplicator(similarity_threshold=0.85, embedding_generator=mock_embedder)
    scores = np.array([0.92, 0.88, 0.40], dtype=np.float32)

    res_df, res_scores, stats = dedup.deduplicate(
        df, similarity_scores=scores, representative_strategy="longest"
    )

    assert len(res_df) == 2
    assert stats["total_input_count"] == 3
    assert stats["unique_count"] == 2
    assert stats["duplicate_count"] == 1
    assert stats["semantic_duplicates_count"] == 1
    assert stats["exact_duplicates_count"] == 0
    assert len(stats["duplicate_groups"]) == 1


def test_representative_selection_strategies():
    mock_embedder = MagicMock()
    # 2 tickets that are semantic duplicates
    df = pd.DataFrame(
        [
            {
                "ticket_id": "T1",
                "topic": "Search Bug",
                "message": "Search is slow",
                "created_at": "2026-01-01 10:00:00",
            },
            {
                "ticket_id": "T2",
                "topic": "Search Bug",
                "message": "Search takes 10 seconds to load when querying large documents",
                "created_at": "2026-01-05 10:00:00",
            },
        ]
    )
    # Identical mock embeddings so they are 100% duplicate
    mock_embedder.encode_texts.return_value = np.array(
        [[1.0, 0.0], [1.0, 0.0]], dtype=np.float32
    )

    dedup = EvidenceDeduplicator(similarity_threshold=0.85, embedding_generator=mock_embedder)
    scores = np.array([0.70, 0.95], dtype=np.float32)

    # Strategy 1: longest -> T2
    res_df_longest, _, _ = dedup.deduplicate(
        df, similarity_scores=scores, representative_strategy="longest"
    )
    assert res_df_longest.iloc[0]["ticket_id"] == "T2"

    # Strategy 2: highest_similarity -> T2 (0.95 > 0.70)
    res_df_sim, _, _ = dedup.deduplicate(
        df, similarity_scores=scores, representative_strategy="highest_similarity"
    )
    assert res_df_sim.iloc[0]["ticket_id"] == "T2"

    # Strategy 3: earliest -> T1 (Jan 1 < Jan 5)
    res_df_early, _, _ = dedup.deduplicate(
        df, similarity_scores=scores, representative_strategy="earliest"
    )
    assert res_df_early.iloc[0]["ticket_id"] == "T1"
