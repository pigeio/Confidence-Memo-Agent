import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.memo_service import MemoService
from src.retrieval import retrieve_tickets
from src.evidence_validation import validate_retrieved_evidence
from src.evidence_scoring import EvidenceScoringEngine
from src.config import EVIDENCE_SIMILARITY_THRESHOLD


def test_false_positive_retrieval_offline_mode():
    """
    Regression Test:
    Verify that an irrelevant candidate ticket (e.g. Dark Mode ticket retrieved for an 'Offline Mode' query)
    is filtered out by Evidence Validation (similarity < 0.60), producing:
    - 0 validated tickets
    - Evidence Score = 0
    - Confidence Level = Low
    - No build or reject recommendation from Gemini
    """
    # 1. Dataset containing only an off-topic Dark Mode ticket
    df = pd.DataFrame([
        {
            "ticket_id": 1,
            "topic": "Dark Mode support",
            "message": "The screen is too bright at night. Can we get dark mode? It hurts my eyes.",
        }
    ])

    keywords = ["offline mode", "no internet connection"]
    proposal = "Should we build Offline Mode?"

    # Low similarity score (e.g., 0.21) for the off-topic ticket
    low_scores = np.array([0.21], dtype=np.float32)

    # 2. Evidence Validation step removes the low-similarity ticket
    validated_tickets, validated_scores = validate_retrieved_evidence(
        df, low_scores, threshold=EVIDENCE_SIMILARITY_THRESHOLD, return_scores=True
    )

    assert validated_tickets.empty
    assert len(validated_scores) == 0

    # 3. Evidence Scoring Engine produces 0 score and Low confidence
    scoring_engine = EvidenceScoringEngine()
    scoring_info = scoring_engine.calculate_score(
        validated_tickets, similarity_scores=validated_scores, total_retrieved_count=len(df)
    )

    assert scoring_info["score"] == 0
    assert scoring_info["confidence"] == "Low"
    assert scoring_info["evidence_summary"]["validated_tickets"] == 0
    assert scoring_info["evidence_summary"]["retrieved_tickets"] == 1
    assert scoring_info["evidence_summary"]["rejected_tickets"] == 1

    # 4. Orchestration via MemoService produces low confidence response
    mock_gemini = MagicMock()
    mock_gemini.generate_response.return_value = (
        "Confidence: Low (Score: 0 / 100)\n\n"
        "Evidence:\n"
        "- No validated customer evidence was found for this proposal.\n\n"
        "Missing:\n"
        "- Customer support tickets, telemetry analytics, and user research.\n\n"
        "Recommendation:\n"
        "- There is currently insufficient validated evidence to support or reject this proposal. "
        "We recommend collecting additional customer feedback before prioritization."
    )

    memo_service = MemoService(gemini_client=mock_gemini)
    memo = memo_service.generate_evidence_memo(df, keywords, proposal)

    assert "Confidence: Low" in memo
    assert "No validated customer evidence" in memo
    assert "insufficient validated evidence to support or reject" in memo
