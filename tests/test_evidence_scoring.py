import pytest
import pandas as pd
from src.evidence_scoring import EvidenceScoringEngine


@pytest.fixture
def engine():
    """Pytest fixture providing an instance of EvidenceScoringEngine."""
    return EvidenceScoringEngine()


@pytest.fixture
def single_ticket_df():
    """Fixture providing a DataFrame with a single support ticket."""
    return pd.DataFrame([
        {
            "ticket_id": 1,
            "topic": "Feature request",
            "message": "It would be nice to have dark mode."
        }
    ])


@pytest.fixture
def high_evidence_df():
    """Fixture providing a DataFrame with multiple urgent support tickets."""
    return pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "The screen hurts my eyes at night. Urgent crash issue.", "created_at": "2026-08-01"},
        {"ticket_id": 2, "topic": "Eye Strain", "message": "Severe eye strain when using the app. Blinding white background.", "created_at": "2026-08-02"},
        {"ticket_id": 3, "topic": "Dark Theme", "message": "Please add dark mode! Eyes are burning.", "created_at": "2026-08-03"},
        {"ticket_id": 4, "topic": "Glare problem", "message": "App crashes under glare. Can't read text.", "created_at": "2026-08-04"},
        {"ticket_id": 5, "topic": "Night mode", "message": "Cannot work at night without dark theme. Urgent fix needed.", "created_at": "2026-08-05"}
    ])


# 1. Empty DataFrame
def test_empty_dataframe(engine):
    """Verify that an empty DataFrame returns a score of 0 and Low confidence."""
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    result = engine.calculate_score(empty_df)

    assert result["score"] == 0
    assert result["confidence"] == "Low"
    assert result["factors"]["ticket_volume"] == 0.0
    assert result["factors"]["severity"] == 0.0
    assert result["factors"]["sentiment_consistency"] == 0.0


# 2. Single ticket evaluation (capped at Low confidence / max 39)
def test_single_ticket(engine, single_ticket_df):
    """Verify scoring logic and small dataset cap (max 39) for a single ticket dataset."""
    result = engine.calculate_score(single_ticket_df)

    assert isinstance(result["score"], int)
    assert 0 <= result["score"] <= 39
    assert result["confidence"] == "Low"
    assert result["factors"]["ticket_volume"] == 6.0
    assert result["evidence_summary"]["sample_size"] == "Small"


# 3. Multiple tickets evaluation (High Evidence)
def test_multiple_tickets_high_evidence(engine, high_evidence_df):
    """Verify scoring logic for multiple tickets with high severity and volume."""
    result = engine.calculate_score(high_evidence_df)

    assert result["score"] >= 70
    assert result["confidence"] == "High"
    assert result["factors"]["ticket_volume"] == 30.0
    assert result["factors"]["severity"] >= 16.0


# 4. Confidence mapping boundaries (0-39 Low, 40-69 Moderate, 70-100 High)
def test_confidence_mapping_boundaries(engine):
    """Verify confidence level mapping across boundary scores."""
    assert engine._map_confidence_level(0) == "Low"
    assert engine._map_confidence_level(39) == "Low"
    assert engine._map_confidence_level(40) == "Moderate"
    assert engine._map_confidence_level(69) == "Moderate"
    assert engine._map_confidence_level(70) == "High"
    assert engine._map_confidence_level(100) == "High"


# 5. Invalid input types
def test_invalid_input_type_raises_error(engine):
    """Verify passing a non-DataFrame input raises TypeError."""
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        engine.calculate_score("not_a_dataframe")

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        engine.calculate_score([1, 2, 3])


# 6. Factor breakdown & evidence_summary structure
def test_factor_breakdown_structure(engine, single_ticket_df):
    """Verify that all required factor keys and evidence_summary exist in the returned dictionary."""
    result = engine.calculate_score(single_ticket_df)

    assert "score" in result
    assert "confidence" in result
    assert "factors" in result
    assert "evidence_summary" in result

    summary = result["evidence_summary"]
    expected_summary_keys = {
        "validated_tickets",
        "retrieved_tickets",
        "rejected_tickets",
        "average_similarity",
        "sample_size",
    }
    assert set(summary.keys()) == expected_summary_keys

    factors = result["factors"]
    expected_factor_keys = {
        "ticket_volume",
        "severity",
        "sentiment_consistency",
        "recency",
        "diversity",
    }
    assert set(factors.keys()) == expected_factor_keys


# 7. Deterministic output
def test_deterministic_output(engine, high_evidence_df):
    """Verify that calculate_score returns identical outputs when called repeatedly."""
    res1 = engine.calculate_score(high_evidence_df)
    res2 = engine.calculate_score(high_evidence_df)

    assert res1 == res2


# 8. Maximum and minimum score bounds
def test_score_bounding_limits(engine):
    """Verify scores never exceed 100 or fall below 0."""
    empty_df = pd.DataFrame(columns=["topic", "message"])
    assert engine.calculate_score(empty_df)["score"] == 0

    # Test large dataset
    large_df = pd.DataFrame([
        {
            "ticket_id": i,
            "topic": f"Crash {i}",
            "message": "Urgent severe crash hurting eyes burning blinding!",
            "created_at": "2026-08-01"
        }
        for i in range(50)
    ])
    result = engine.calculate_score(large_df)
    assert result["score"] == 100
    assert result["confidence"] == "High"


# 9. Small dataset score caps (1 ticket <= 39, 2 tickets <= 69)
def test_small_dataset_score_caps(engine):
    """Verify that 1 ticket is capped at 39 (Low) and 2 tickets are capped at 69 (Moderate)."""
    # 1 urgent ticket with max severity/recency
    df1 = pd.DataFrame([
        {"ticket_id": 1, "topic": "Crash", "message": "Urgent severe crash hurting eyes!", "created_at": "2026-08-01"}
    ])
    res1 = engine.calculate_score(df1)
    assert res1["score"] <= 39
    assert res1["confidence"] == "Low"

    # 2 urgent tickets
    df2 = pd.DataFrame([
        {"ticket_id": 1, "topic": "Crash 1", "message": "Urgent severe crash hurting eyes!", "created_at": "2026-08-01"},
        {"ticket_id": 2, "topic": "Crash 2", "message": "Urgent severe crash burning eyes!", "created_at": "2026-08-02"}
    ])
    res2 = engine.calculate_score(df2)
    assert res2["score"] <= 69
    assert res2["confidence"] in ["Low", "Moderate"]


# 10. Similarity score weighting
def test_similarity_score_weighting(engine):
    """Verify that higher similarity scores increase weighted factor scores."""
    df = pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "Urgent dark mode feature needed.", "created_at": "2026-08-01"},
        {"ticket_id": 2, "topic": "Eye Strain", "message": "Urgent eye strain fix needed.", "created_at": "2026-08-02"},
        {"ticket_id": 3, "topic": "Dark Theme", "message": "Urgent dark theme fix needed.", "created_at": "2026-08-03"}
    ])

    import numpy as np
    high_sims = np.array([0.95, 0.92, 0.98], dtype=np.float32)
    low_sims = np.array([0.61, 0.60, 0.62], dtype=np.float32)

    res_high = engine.calculate_score(df, similarity_scores=high_sims)
    res_low = engine.calculate_score(df, similarity_scores=low_sims)

    assert res_high["score"] > res_low["score"]
    assert res_high["factors"]["ticket_volume"] > res_low["factors"]["ticket_volume"]
    assert res_high["evidence_summary"]["average_similarity"] > res_low["evidence_summary"]["average_similarity"]

