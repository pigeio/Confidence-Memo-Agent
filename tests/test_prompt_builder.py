import pytest
import pandas as pd
from src.prompt_builder import build_evidence_prompt, format_tickets


@pytest.fixture
def sample_proposal() -> str:
    """Fixture providing a standard feature proposal string."""
    return "Add dark mode support"


@pytest.fixture
def sample_tickets_df() -> pd.DataFrame:
    """Fixture providing a small synthetic DataFrame of support tickets."""
    return pd.DataFrame([
        {
            "ticket_id": 101,
            "topic": "Dark Mode Request",
            "message": "Please add a dark theme option."
        },
        {
            "ticket_id": 102,
            "topic": "CSV Export",
            "message": "We need CSV download functionality."
        }
    ])


# --- format_tickets Unit Tests ---

def test_format_tickets_empty():
    """Verify format_tickets returns fallback message for empty DataFrames."""
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    result = format_tickets(empty_df)
    assert result == "No tickets found matching the keyword criteria."


def test_format_tickets_valid_structure(sample_tickets_df):
    """Verify format_tickets formats valid ticket rows with ID, topic, and message."""
    result = format_tickets(sample_tickets_df)
    assert "Ticket ID: 101" in result
    assert "Topic: Dark Mode Request" in result
    assert "Message: Please add a dark theme option." in result
    assert "---" in result


def test_format_tickets_preserves_order():
    """Verify format_tickets preserves the exact order of rows from the DataFrame."""
    ordered_df = pd.DataFrame([
        {"ticket_id": 1, "topic": "First Ticket", "message": "Alpha"},
        {"ticket_id": 2, "topic": "Second Ticket", "message": "Beta"},
        {"ticket_id": 3, "topic": "Third Ticket", "message": "Gamma"},
    ])
    result = format_tickets(ordered_df)
    
    pos_first = result.find("Ticket ID: 1")
    pos_second = result.find("Ticket ID: 2")
    pos_third = result.find("Ticket ID: 3")
    
    assert pos_first != -1 and pos_second != -1 and pos_third != -1
    assert pos_first < pos_second < pos_third


# --- format_scoring_summary Unit Tests ---

def test_format_scoring_summary_with_telemetry():
    """Verify format_scoring_summary includes telemetry stats (validated, retrieved, rejected, avg similarity)."""
    from src.prompt_builder import format_scoring_summary

    scoring_info = {
        "score": 75,
        "confidence": "High",
        "evidence_summary": {
            "validated_tickets": 5,
            "retrieved_tickets": 8,
            "rejected_tickets": 3,
            "average_similarity": 0.81,
            "sample_size": "Medium",
        },
        "factors": {
            "ticket_volume": 30.0,
            "severity": 16.0,
            "sentiment_consistency": 25.0,
            "recency": 15.0,
            "diversity": 10.0,
        },
    }

    result = format_scoring_summary(scoring_info)
    assert "Confidence Level: High" in result
    assert "Evidence Score: 75 / 100" in result
    assert "Evidence Telemetry:" in result
    assert "Validated Tickets: 5" in result
    assert "Retrieved Candidates: 8" in result
    assert "Rejected Candidates: 3" in result
    assert "Average Similarity: 0.81" in result
    assert "Sample Size Tier: Medium" in result
    assert "Factor Breakdown:" in result


# --- build_evidence_prompt Unit Tests ---


def test_build_prompt_contains_proposal(sample_proposal, sample_tickets_df):
    """Verify build_evidence_prompt incorporates the feature proposal into the prompt."""
    prompt = build_evidence_prompt(sample_proposal, sample_tickets_df)
    assert "Add dark mode support" in prompt


def test_build_prompt_contains_ticket_info(sample_proposal, sample_tickets_df):
    """Verify build_evidence_prompt incorporates formatted ticket information."""
    prompt = build_evidence_prompt(sample_proposal, sample_tickets_df)
    assert "Ticket ID: 101" in prompt
    assert "Ticket ID: 102" in prompt


def test_build_prompt_full_placeholder_replacement(sample_proposal, sample_tickets_df):
    """Verify all template placeholders ({proposal}, {tickets_text}) are completely substituted."""
    prompt = build_evidence_prompt(sample_proposal, sample_tickets_df)
    assert "{proposal}" not in prompt
    assert "{tickets_text}" not in prompt


def test_build_prompt_empty_tickets_df(sample_proposal):
    """Verify an empty ticket DataFrame produces a valid prompt containing the proposal."""
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    prompt = build_evidence_prompt(sample_proposal, empty_df)
    
    assert "Add dark mode support" in prompt
    assert "No tickets found matching the keyword criteria." in prompt
    assert "{proposal}" not in prompt
    assert "{tickets_text}" not in prompt


def test_build_prompt_unicode_support():
    """Verify build_evidence_prompt correctly handles Unicode characters and emojis in proposal and tickets."""
    unicode_proposal = "Enable Mode Sombre 🌙 / Dark Theme (Español 🇪🇸 & 日本語 🇯🇵)"
    unicode_df = pd.DataFrame([
        {
            "ticket_id": 999,
            "topic": "Problème d'éclairage 💡",
            "message": "La pantalla es demasiado brillante 🌟. Necesitamos modo oscuro!"
        }
    ])
    
    prompt = build_evidence_prompt(unicode_proposal, unicode_df)
    assert "Enable Mode Sombre 🌙 / Dark Theme (Español 🇪🇸 & 日本語 🇯🇵)" in prompt
    assert "Problème d'éclairage 💡" in prompt
    assert "La pantalla es demasiado brillante 🌟. Necesitamos modo oscuro!" in prompt


def test_build_prompt_custom_template_file(tmp_path, sample_proposal, sample_tickets_df):
    """Verify custom template file loading using pytest's tmp_path fixture."""
    custom_template_content = (
        "CUSTOM ANALYSIS PROMPT\n"
        "Target Feature: {proposal}\n"
        "Data Source:\n{tickets_text}\n"
        "END PROMPT"
    )
    template_file = tmp_path / "custom_template.txt"
    template_file.write_text(custom_template_content, encoding="utf-8")

    prompt = build_evidence_prompt(sample_proposal, sample_tickets_df, template_path=str(template_file))
    
    assert prompt.startswith("CUSTOM ANALYSIS PROMPT")
    assert "Target Feature: Add dark mode support" in prompt
    assert "Ticket ID: 101" in prompt
    assert prompt.endswith("END PROMPT")


# --- Error Handling & Input Validation Tests ---

def test_build_prompt_invalid_proposal_type(sample_tickets_df):
    """Verify passing non-string proposal raises ValueError."""
    with pytest.raises(ValueError, match="proposal must be a non-empty string"):
        build_evidence_prompt(12345, sample_tickets_df)


def test_build_prompt_empty_proposal(sample_tickets_df):
    """Verify passing empty or whitespace proposal raises ValueError."""
    with pytest.raises(ValueError, match="proposal must be a non-empty string"):
        build_evidence_prompt("", sample_tickets_df)
    with pytest.raises(ValueError, match="proposal must be a non-empty string"):
        build_evidence_prompt("   ", sample_tickets_df)


def test_build_prompt_invalid_df_type(sample_proposal):
    """Verify passing non-DataFrame raises TypeError."""
    with pytest.raises(TypeError, match="df_tickets must be a pandas DataFrame"):
        build_evidence_prompt(sample_proposal, "not_a_dataframe")


def test_build_prompt_df_missing_columns(sample_proposal):
    """Verify DataFrame missing required columns ('topic', 'message') raises ValueError."""
    invalid_df = pd.DataFrame({"ticket_id": [1], "user_comment": ["Hello"]})
    with pytest.raises(ValueError, match="df_tickets is missing required columns"):
        build_evidence_prompt(sample_proposal, invalid_df)


def test_build_prompt_missing_template_file(sample_proposal, sample_tickets_df):
    """Verify specifying a non-existent template path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Template file not found"):
        build_evidence_prompt(sample_proposal, sample_tickets_df, template_path="non_existent_file.txt")


def test_build_prompt_bad_template_placeholders(tmp_path, sample_proposal, sample_tickets_df):
    """Verify custom template missing expected placeholders raises KeyError."""
    bad_template = tmp_path / "bad_template.txt"
    bad_template.write_text("Template with invalid {wrong_key} placeholder", encoding="utf-8")

    with pytest.raises(KeyError, match="Template is missing required placeholder"):
        build_evidence_prompt(sample_proposal, sample_tickets_df, template_path=str(bad_template))


# --- build_decision_prompt Unit Tests ---

def test_build_decision_prompt_structure(sample_proposal, sample_tickets_df):
    from src.prompt_builder import build_decision_prompt

    decision_info = {
        "recommendation": "PROCEED_TO_BUILD",
        "priority_score": 85,
        "decision_tier": "Top Priority",
        "evidence_score": 80,
        "assumptions": ["Assumes engineering capacity available."],
        "risks": ["Low execution risk."],
        "trade_offs": ["High ROI vs moderate effort."],
        "missing_information": ["None."],
        "rationale": ["Solid customer evidence and high priority score."],
    }
    scoring_info = {"score": 80, "confidence": "High"}
    dedup_stats = {
        "total_input_count": 5,
        "unique_count": 3,
        "duplicate_count": 2,
        "duplicate_rate": 0.4,
        "exact_duplicates_count": 1,
        "semantic_duplicates_count": 1,
    }
    clustering_stats = {
        "total_clusters": 2,
        "average_cluster_size": 1.5,
        "clusters": [{"cluster_id": 0, "size": 2, "theme_label": "Theme A"}],
    }

    prompt = build_decision_prompt(
        proposal=sample_proposal,
        df_tickets=sample_tickets_df,
        decision_info=decision_info,
        scoring_info=scoring_info,
        deduplication_stats=dedup_stats,
        clustering_stats=clustering_stats,
    )

    assert "Decision Memo" in prompt
    assert "PROCEED_TO_BUILD" in prompt
    assert "Priority Score: 85 / 100" in prompt
    assert "Deduplication Telemetry:" in prompt
    assert "Clustering & Theme Analysis:" in prompt
    assert "Assumes engineering capacity available." in prompt
