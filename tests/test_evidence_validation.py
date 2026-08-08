import pytest
import numpy as np
import pandas as pd
from src.evidence_validation import validate_retrieved_evidence


@pytest.fixture
def sample_tickets():
    """Fixture providing a small DataFrame of retrieved tickets."""
    return pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "Need dark theme"},
        {"ticket_id": 2, "topic": "CSV Export", "message": "Download CSV data"},
        {"ticket_id": 3, "topic": "Bug Report", "message": "App crashes on login"},
        {"ticket_id": 4, "topic": "Feedback", "message": "Great product overall"},
    ])


# 1. Empty DataFrame
def test_empty_dataframe():
    """Verify that an empty DataFrame returns an empty DataFrame."""
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    scores = np.array([], dtype=np.float32)

    result = validate_retrieved_evidence(empty_df, scores, threshold=0.5)

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# 2. All tickets below threshold
def test_all_below_threshold(sample_tickets):
    """Verify that when all scores are below threshold, an empty DataFrame is returned."""
    scores = np.array([0.10, 0.20, 0.15, 0.30], dtype=np.float32)

    result = validate_retrieved_evidence(sample_tickets, scores, threshold=0.60)

    assert result.empty
    assert list(result.columns) == list(sample_tickets.columns)


# 3. All tickets above threshold
def test_all_above_threshold(sample_tickets):
    """Verify that when all scores meet threshold, all tickets are returned."""
    scores = np.array([0.90, 0.85, 0.70, 0.65], dtype=np.float32)

    result = validate_retrieved_evidence(sample_tickets, scores, threshold=0.60)

    assert len(result) == 4


# 4. Mixed relevance
def test_mixed_relevance(sample_tickets):
    """Verify that only tickets meeting the threshold survive filtering."""
    scores = np.array([0.85, 0.30, 0.75, 0.10], dtype=np.float32)

    result = validate_retrieved_evidence(sample_tickets, scores, threshold=0.60)

    assert len(result) == 2
    assert list(result["ticket_id"]) == [1, 3]


# 5. Threshold boundary (exact match)
def test_threshold_boundary(sample_tickets):
    """Verify that a score exactly equal to the threshold is included."""
    scores = np.array([0.60, 0.59, 0.61, 0.60], dtype=np.float32)

    result = validate_retrieved_evidence(sample_tickets, scores, threshold=0.60)

    assert len(result) == 3
    assert list(result["ticket_id"]) == [1, 3, 4]


# 6. Preserved ordering
def test_preserved_ordering(sample_tickets):
    """Verify that the original ranking order is preserved after filtering."""
    scores = np.array([0.95, 0.10, 0.80, 0.70], dtype=np.float32)

    result = validate_retrieved_evidence(sample_tickets, scores, threshold=0.60)

    assert list(result["ticket_id"]) == [1, 3, 4]


# 7. Invalid input — tickets not a DataFrame
def test_invalid_tickets_type():
    """Verify that passing non-DataFrame tickets raises TypeError."""
    with pytest.raises(TypeError, match="tickets must be a pandas DataFrame"):
        validate_retrieved_evidence("not_a_dataframe", np.array([0.5]))


# 8. Invalid input — scores not an ndarray
def test_invalid_scores_type(sample_tickets):
    """Verify that passing non-ndarray scores raises TypeError."""
    with pytest.raises(TypeError, match="similarity_scores must be a numpy ndarray"):
        validate_retrieved_evidence(sample_tickets, [0.5, 0.6, 0.7, 0.8])


# 9. Length mismatch between tickets and scores
def test_length_mismatch(sample_tickets):
    """Verify that mismatched lengths raise ValueError."""
    scores = np.array([0.5, 0.6], dtype=np.float32)  # 2 scores for 4 tickets

    with pytest.raises(ValueError, match="Length mismatch"):
        validate_retrieved_evidence(sample_tickets, scores, threshold=0.5)
