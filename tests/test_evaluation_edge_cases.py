import pytest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from src.memo_service import MemoService
from src.evaluation.adapters import GooglePlayAdapter, BaseDatasetAdapter
from src.evidence_deduplication import EvidenceDeduplicator
from src.evidence_clustering import EvidenceClusterer
from src.decision_engine import DecisionEngine


@pytest.fixture
def mock_memo_service():
    mock_client = MagicMock()
    mock_client.generate_response.return_value = "Mocked Memo Response"
    return MemoService(gemini_client=mock_client)


def test_edge_case_missing_timestamps(mock_memo_service):
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "Theme", "message": "Add dark mode", "created_at": None},
            {"ticket_id": 2, "topic": "Theme", "message": "White screen hurts eyes", "created_at": float("nan")},
            {"ticket_id": 3, "topic": "Theme", "message": "Need dark theme", "created_at": "invalid-date-string"},
        ]
    )
    memo, result = mock_memo_service.generate_decision_memo(
        df=df,
        keywords=["dark", "mode", "theme"],
        proposal="Dark Mode Support",
        return_decision_result=True,
    )
    assert isinstance(memo, str)
    assert result.evidence_score > 0


def test_edge_case_missing_topics(mock_memo_service):
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": None, "message": "Search queries fail on large repos"},
            {"ticket_id": 2, "topic": "", "message": "Search filtering is broken"},
            {"ticket_id": 3, "topic": float("nan"), "message": "Search timeout occurs frequently"},
        ]
    )
    # Adapt via standard schema validation
    adapter = GooglePlayAdapter()
    clean_df = adapter.validate_schema(df)

    memo, result = mock_memo_service.generate_decision_memo(
        df=clean_df,
        keywords=["search", "queries", "timeout"],
        proposal="Search Performance",
        return_decision_result=True,
    )
    assert isinstance(memo, str)
    assert result.evidence_score > 0


def test_edge_case_duplicate_ids(mock_memo_service):
    df = pd.DataFrame(
        [
            {"ticket_id": "SAME_ID", "topic": "Theme", "message": "Dark mode please"},
            {"ticket_id": "SAME_ID", "topic": "Theme", "message": "Night theme is essential"},
            {"ticket_id": "SAME_ID", "topic": "Theme", "message": "Too bright at night"},
        ]
    )
    memo, result = mock_memo_service.generate_decision_memo(
        df=df,
        keywords=["dark", "mode", "theme"],
        proposal="Dark Mode Support",
        return_decision_result=True,
    )
    assert isinstance(memo, str)
    assert result.evidence_score > 0


def test_edge_case_extremely_long_message(mock_memo_service):
    long_msg = "Please add dark mode. " * 500  # ~11,500 characters
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "Theme", "message": long_msg},
            {"ticket_id": 2, "topic": "Theme", "message": "Dark theme needed"},
        ]
    )
    memo, result = mock_memo_service.generate_decision_memo(
        df=df,
        keywords=["dark", "mode"],
        proposal="Dark Mode Support",
        return_decision_result=True,
    )
    assert isinstance(memo, str)
    assert result.evidence_score > 0


def test_edge_case_very_short_messages(mock_memo_service):
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "Dark Mode", "message": "k"},
            {"ticket_id": 2, "topic": "Dark Mode", "message": "?"},
            {"ticket_id": 3, "topic": "Dark Mode", "message": "dark"},
        ]
    )
    memo, result = mock_memo_service.generate_decision_memo(
        df=df,
        keywords=["dark", "mode"],
        proposal="Dark Mode Support",
        return_decision_result=True,
    )
    assert isinstance(memo, str)


def test_edge_case_multilingual_feedback(mock_memo_service):
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "Tema", "message": "Por favor agreguen modo oscuro, la pantalla blanca me lastima la vista."},
            {"ticket_id": 2, "topic": "Thème", "message": "Nous avons besoin d'un mode sombre pour l'application."},
            {"ticket_id": 3, "topic": "Design", "message": "Dunkelmodus ist dringend erforderlich für die Nachtarbeit."},
            {"ticket_id": 4, "topic": "UI", "message": "ダークモードの追加をお願いします。白い背景は目が疲れます。"},
        ]
    )
    memo, result = mock_memo_service.generate_decision_memo(
        df=df,
        keywords=["dark", "mode", "sombre", "oscuro", "dunkel", "ダークモード"],
        proposal="Multilingual Dark Theme Support",
        return_decision_result=True,
    )
    assert isinstance(memo, str)
    assert result.recommendation is not None
    assert 0 <= result.priority_score <= 100


def test_edge_case_emojis_and_special_characters(mock_memo_service):
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "🔥 Theme 🌙", "message": "The white UI hurts my eyes!! 😭 Blinding bright ⚡⚡⚡ Please fix 🙏"},
            {"ticket_id": 2, "topic": "SQL; DROP TABLE users;--", "message": "<script>alert('xss')</script> &amp; quote's #1234"},
        ]
    )
    memo, result = mock_memo_service.generate_decision_memo(
        df=df,
        keywords=["theme", "bright", "white", "eyes"],
        proposal="Theme Refinement",
        return_decision_result=True,
    )
    assert isinstance(memo, str)


def test_edge_case_empty_and_single_dataset(mock_memo_service):
    empty_df = pd.DataFrame(columns=["ticket_id", "created_at", "topic", "message"])
    memo_empty, res_empty = mock_memo_service.generate_decision_memo(
        df=empty_df,
        keywords=["anything"],
        proposal="Anything",
        return_decision_result=True,
    )
    assert res_empty.evidence_score == 0

    single_df = pd.DataFrame([{"ticket_id": "1", "topic": "Theme", "message": "Dark mode"}])
    memo_single, res_single = mock_memo_service.generate_decision_memo(
        df=single_df,
        keywords=["dark"],
        proposal="Dark Mode",
        return_decision_result=True,
    )
    assert isinstance(memo_single, str)
