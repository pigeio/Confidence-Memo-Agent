import pytest
from unittest.mock import MagicMock, patch, Mock
import numpy as np
import pandas as pd
from src.memo_service import MemoService


@pytest.fixture
def sample_inputs():
    """Fixture providing valid sample inputs for MemoService."""
    df = pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "Please add dark mode"}
    ])
    keywords = ["dark"]
    proposal = "Implement dark mode theme"
    return df, keywords, proposal


@pytest.fixture
def mock_gemini_client():
    """Fixture providing a mocked GeminiClient instance."""
    client = MagicMock()
    client.generate_response.return_value = "Generated Evidence Memo"
    return client


@pytest.fixture
def high_scores():
    """Fixture providing similarity scores above the evidence threshold."""
    return np.array([0.85], dtype=np.float32)


# 1. Successful end-to-end orchestration
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_successful_end_to_end_orchestration(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify that generate_evidence_memo orchestrates the full pipeline successfully."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = (df, high_scores)
    mock_validate.return_value = (df, high_scores)
    mock_build_prompt.return_value = "Formatted Prompt Text"

    service = MemoService(gemini_client=mock_gemini_client)
    memo = service.generate_evidence_memo(df, keywords, proposal)

    assert memo == "Generated Evidence Memo"


# 2. Correct arguments passed through the pipeline
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_arguments_passed_through_pipeline(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify that retrieval results flow correctly through validation to prompt builder."""
    df_input, keywords, proposal = sample_inputs
    retrieved_df = pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "Matched ticket"}
    ])
    mock_retrieve_tickets.return_value = (retrieved_df, high_scores)
    mock_validate.return_value = (retrieved_df, high_scores)
    mock_build_prompt.return_value = "Prompt String"

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df_input, keywords, proposal, template_path="custom.txt")

    mock_retrieve_tickets.assert_called_once_with(df_input, keywords)
    mock_validate.assert_called_once_with(retrieved_df, high_scores, return_scores=True)
    assert mock_build_prompt.call_count == 1
    args, kwargs = mock_build_prompt.call_args
    assert kwargs["template_path"] == "custom.txt"
    assert "scoring_info" in kwargs


# 3. Correct prompt passed to Gemini
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_correct_prompt_passed_to_gemini(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify that the exact prompt output by build_evidence_prompt is sent to Gemini."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = (df, high_scores)
    mock_validate.return_value = (df, high_scores)
    expected_prompt = "EXPECTED ANALYST PROMPT CONTENT"
    mock_build_prompt.return_value = expected_prompt

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df, keywords, proposal)

    mock_gemini_client.generate_response.assert_called_once_with(expected_prompt)


# 4. Empty retrieval results produce a memo with score=0
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_empty_retrieval_results_produce_no_evidence_memo(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client
):
    """Verify that empty retrieval produces score=0, confidence=Low with a reason."""
    df, keywords, proposal = sample_inputs
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    empty_scores = np.array([], dtype=np.float32)
    mock_retrieve_tickets.return_value = (empty_df, empty_scores)
    mock_validate.return_value = (empty_df, empty_scores)
    mock_build_prompt.return_value = "Prompt for no evidence"
    mock_gemini_client.generate_response.return_value = "No evidence memo"

    service = MemoService(gemini_client=mock_gemini_client)
    memo = service.generate_evidence_memo(df, keywords, proposal)

    assert memo == "No evidence memo"
    args, kwargs = mock_build_prompt.call_args
    assert kwargs["scoring_info"]["score"] == 0
    assert kwargs["scoring_info"]["confidence"] == "Low"


# 5. Validation filters out irrelevant tickets → score=0 short-circuit
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_validation_filters_all_tickets_produces_no_evidence(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client
):
    """Verify that when validation filters out ALL tickets, scoring is skipped and score=0."""
    df, keywords, proposal = sample_inputs
    low_scores = np.array([0.15], dtype=np.float32)
    mock_retrieve_tickets.return_value = (df, low_scores)
    # Validation returns empty because all scores are below threshold
    empty_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    empty_scores = np.array([], dtype=np.float32)
    mock_validate.return_value = (empty_df, empty_scores)
    mock_build_prompt.return_value = "Prompt"

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df, keywords, proposal)

    args, kwargs = mock_build_prompt.call_args
    assert kwargs["scoring_info"]["score"] == 0
    assert kwargs["scoring_info"]["confidence"] == "Low"


# 6. Retrieval exceptions propagate correctly
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_retrieval_exceptions_propagate(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client
):
    """Verify that exceptions raised by retrieval propagate and stop downstream execution."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.side_effect = ValueError("DataFrame is missing required columns")

    service = MemoService(gemini_client=mock_gemini_client)

    with pytest.raises(ValueError, match="missing required columns"):
        service.generate_evidence_memo(df, keywords, proposal)

    mock_validate.assert_not_called()
    mock_build_prompt.assert_not_called()
    mock_gemini_client.generate_response.assert_not_called()


# 7. Prompt builder exceptions propagate correctly
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_prompt_builder_exceptions_propagate(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify that exceptions raised by prompt_builder propagate and prevent calling Gemini."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = (df, high_scores)
    mock_validate.return_value = (df, high_scores)
    mock_build_prompt.side_effect = KeyError("Missing required template placeholder")

    service = MemoService(gemini_client=mock_gemini_client)

    with pytest.raises(KeyError, match="Missing required template placeholder"):
        service.generate_evidence_memo(df, keywords, proposal)

    mock_gemini_client.generate_response.assert_not_called()


# 8. Gemini exceptions propagate correctly
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_gemini_exceptions_propagate(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify that exceptions from GeminiClient propagate out of generate_evidence_memo."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = (df, high_scores)
    mock_validate.return_value = (df, high_scores)
    mock_build_prompt.return_value = "Prompt Text"
    mock_gemini_client.generate_response.side_effect = RuntimeError("Gemini API connection error")

    service = MemoService(gemini_client=mock_gemini_client)

    with pytest.raises(RuntimeError, match="Gemini API connection error"):
        service.generate_evidence_memo(df, keywords, proposal)


# 9. Verify each dependency is called exactly once during a successful execution
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_dependencies_called_exactly_once(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify each pipeline dependency is invoked exactly once."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = (df, high_scores)
    mock_validate.return_value = (df, high_scores)
    mock_build_prompt.return_value = "Prompt Text"

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df, keywords, proposal)

    assert mock_retrieve_tickets.call_count == 1
    assert mock_validate.call_count == 1
    assert mock_build_prompt.call_count == 1
    assert mock_gemini_client.generate_response.call_count == 1


# 10. Verify the final return value is exactly the Gemini response
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_final_return_value_is_gemini_response(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client, high_scores
):
    """Verify that the return value of generate_evidence_memo is identical to Gemini's output string."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = (df, high_scores)
    mock_validate.return_value = (df, high_scores)
    mock_build_prompt.return_value = "Prompt"
    exact_gemini_response = "UNIQUE_MEMO_STRING_RESPONSE_12345"
    mock_gemini_client.generate_response.return_value = exact_gemini_response

    service = MemoService(gemini_client=mock_gemini_client)
    result = service.generate_evidence_memo(df, keywords, proposal)

    assert result is exact_gemini_response


# 11. Verify pre-validated tickets parameter skips retrieval and validation steps
@patch("src.memo_service.validate_retrieved_evidence")
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_pre_validated_tickets_bypasses_retrieval(
    mock_build_prompt, mock_retrieve_tickets, mock_validate,
    sample_inputs, mock_gemini_client
):
    """Verify that passing pre-validated tickets skips retrieve_tickets and validate_retrieved_evidence."""
    df, keywords, proposal = sample_inputs
    mock_build_prompt.return_value = "Prompt"
    mock_gemini_client.generate_response.return_value = "Pre-validated Memo"

    service = MemoService(gemini_client=mock_gemini_client)
    result = service.generate_evidence_memo(
        df, keywords, proposal, validated_tickets=df
    )

    assert mock_retrieve_tickets.call_count == 0
    assert mock_validate.call_count == 0
    assert mock_build_prompt.call_count == 1
    assert result == "Pre-validated Memo"


