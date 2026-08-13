import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.memo_service import MemoService
from src.decision_engine import DecisionResult, RecommendationAction
from src.historical_calibration import InMemoryCalibrationStorage, HistoricalCalibrationEngine


@pytest.fixture
def mock_gemini():
    client = MagicMock()
    client.generate_response.return_value = (
        "# Decision Memo: Dark Mode Support\n\n"
        "**Executive Recommendation:** PROCEED_TO_BUILD\n"
        "**Priority Score:** 85 / 100 (Top Priority)\n"
        "**Evidence Score:** 80 / 100 (High)\n\n"
        "### 1. Recommendation Rationale\n- Strong evidence and high user demand."
    )
    return client


@pytest.fixture
def sample_tickets_df():
    return pd.DataFrame(
        [
            {"ticket_id": 101, "topic": "Dark Mode", "message": "The white UI hurts my eyes at night."},
            {"ticket_id": 102, "topic": "Dark Mode", "message": "Please implement dark mode theme."},
            {"ticket_id": 103, "topic": "Dark Mode", "message": "Night mode is badly needed for developers."},
            {"ticket_id": 104, "topic": "Dark Mode", "message": "Night mode is badly needed for developers."},  # duplicate
        ]
    )


def test_generate_decision_memo_e2e(mock_gemini, sample_tickets_df):
    calibration_storage = InMemoryCalibrationStorage()
    calibration_engine = HistoricalCalibrationEngine(storage=calibration_storage)

    memo_service = MemoService(
        gemini_client=mock_gemini,
        calibration_engine=calibration_engine,
    )

    memo = memo_service.generate_decision_memo(
        df=sample_tickets_df,
        keywords=["dark", "mode", "night", "theme"],
        proposal="Dark Mode Theme Support",
        engineering_effort="Low",
        business_impact="High",
        strategic_alignment="High",
        cost="Low",
        risk="Low",
        record_calibration=True,
    )

    assert isinstance(memo, str)
    assert "Decision Memo" in memo
    mock_gemini.generate_response.assert_called_once()

    # Verify calibration record was saved
    records = calibration_storage.get_all_records()
    assert len(records) == 1
    assert records[0].proposal == "Dark Mode Theme Support"
    assert records[0].predicted_confidence in {"Low", "Moderate", "High"}


def test_generate_decision_memo_return_decision_result(mock_gemini, sample_tickets_df):
    memo_service = MemoService(gemini_client=mock_gemini)

    memo, decision_result = memo_service.generate_decision_memo(
        df=sample_tickets_df,
        keywords=["dark", "mode"],
        proposal="Dark Mode Support",
        engineering_effort="Medium",
        business_impact="Critical",
        return_decision_result=True,
    )

    assert isinstance(memo, str)
    assert isinstance(decision_result, DecisionResult)
    assert decision_result.recommendation in [r.value for r in RecommendationAction]
    assert 0 <= decision_result.priority_score <= 100
    assert len(decision_result.assumptions) > 0
    assert len(decision_result.risks) > 0
    assert len(decision_result.trade_offs) > 0
    assert len(decision_result.missing_information) > 0


def test_generate_decision_memo_zero_evidence(mock_gemini):
    empty_df = pd.DataFrame(columns=["topic", "message"])
    memo_service = MemoService(gemini_client=mock_gemini)

    memo, decision_result = memo_service.generate_decision_memo(
        df=empty_df,
        keywords=["nonexistent"],
        proposal="Completely Unknown Feature",
        return_decision_result=True,
    )

    assert decision_result.recommendation == RecommendationAction.INSUFFICIENT_EVIDENCE.value
    assert decision_result.evidence_score == 0
