import pytest
from unittest.mock import MagicMock, patch, Mock
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


# 1. Successful end-to-end orchestration
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_successful_end_to_end_orchestration(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that generate_evidence_memo orchestrates the full pipeline successfully."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    mock_build_prompt.return_value = "Formatted Prompt Text"

    service = MemoService(gemini_client=mock_gemini_client)
    memo = service.generate_evidence_memo(df, keywords, proposal)

    assert memo == "Generated Evidence Memo"


# 2. Correct arguments passed from retrieval to prompt builder
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_arguments_passed_from_retrieval_to_prompt_builder(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that the DataFrame returned by retrieval is passed directly to prompt_builder."""
    df_input, keywords, proposal = sample_inputs
    retrieved_subset_df = pd.DataFrame([
        {"ticket_id": 1, "topic": "Dark Mode", "message": "Matched ticket"}
    ])
    mock_retrieve_tickets.return_value = retrieved_subset_df
    mock_build_prompt.return_value = "Prompt String"

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df_input, keywords, proposal, template_path="custom.txt")

    mock_retrieve_tickets.assert_called_once_with(df_input, keywords)
    assert mock_build_prompt.call_count == 1
    args, kwargs = mock_build_prompt.call_args
    assert args == (proposal, retrieved_subset_df)
    assert kwargs["template_path"] == "custom.txt"
    assert "scoring_info" in kwargs


# 3. Correct prompt passed to Gemini
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_correct_prompt_passed_to_gemini(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that the exact prompt output by build_evidence_prompt is sent to Gemini."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    expected_prompt = "EXPECTED ANALYST PROMPT CONTENT"
    mock_build_prompt.return_value = expected_prompt

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df, keywords, proposal)

    mock_gemini_client.generate_response.assert_called_once_with(expected_prompt)


# 4. Empty retrieval results still produce a memo
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_empty_retrieval_results_produce_memo(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that an empty retrieval result still proceeds to prompt building and Gemini execution."""
    df, keywords, proposal = sample_inputs
    empty_retrieved_df = pd.DataFrame(columns=["ticket_id", "topic", "message"])
    mock_retrieve_tickets.return_value = empty_retrieved_df
    mock_build_prompt.return_value = "Prompt for empty evidence base"
    mock_gemini_client.generate_response.return_value = "Memo with Low Confidence"

    service = MemoService(gemini_client=mock_gemini_client)
    memo = service.generate_evidence_memo(df, keywords, proposal)

    assert memo == "Memo with Low Confidence"
    assert mock_build_prompt.call_count == 1
    args, kwargs = mock_build_prompt.call_args
    assert args == (proposal, empty_retrieved_df)
    assert kwargs["template_path"] is None
    assert kwargs["scoring_info"]["score"] == 0
    assert kwargs["scoring_info"]["confidence"] == "Low"
    mock_gemini_client.generate_response.assert_called_once_with("Prompt for empty evidence base")


# 5. Retrieval exceptions propagate correctly
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_retrieval_exceptions_propagate(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that exceptions raised by retrieval propagate and stop downstream execution."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.side_effect = ValueError("DataFrame is missing required columns")

    service = MemoService(gemini_client=mock_gemini_client)

    with pytest.raises(ValueError, match="missing required columns"):
        service.generate_evidence_memo(df, keywords, proposal)

    mock_build_prompt.assert_not_called()
    mock_gemini_client.generate_response.assert_not_called()


# 6. Prompt builder exceptions propagate correctly
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_prompt_builder_exceptions_propagate(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that exceptions raised by prompt_builder propagate and prevent calling Gemini."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    mock_build_prompt.side_effect = KeyError("Missing required template placeholder")

    service = MemoService(gemini_client=mock_gemini_client)

    with pytest.raises(KeyError, match="Missing required template placeholder"):
        service.generate_evidence_memo(df, keywords, proposal)

    mock_gemini_client.generate_response.assert_not_called()


# 7. Gemini exceptions propagate correctly
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_gemini_exceptions_propagate(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that exceptions from GeminiClient propagate out of generate_evidence_memo."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    mock_build_prompt.return_value = "Prompt Text"
    mock_gemini_client.generate_response.side_effect = RuntimeError("Gemini API connection error")

    service = MemoService(gemini_client=mock_gemini_client)

    with pytest.raises(RuntimeError, match="Gemini API connection error"):
        service.generate_evidence_memo(df, keywords, proposal)


# 8. Verify each dependency is called exactly once during a successful execution
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_dependencies_called_exactly_once(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify each pipeline dependency (retrieval, prompt_builder, gemini) is invoked exactly once."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    mock_build_prompt.return_value = "Prompt Text"

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df, keywords, proposal)

    assert mock_retrieve_tickets.call_count == 1
    assert mock_build_prompt.call_count == 1
    assert mock_gemini_client.generate_response.call_count == 1


# 9. Verify dependency call order: retrieval → prompt_builder → gemini_client
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_dependency_call_order(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify strict execution order: 1. retrieve_tickets -> 2. build_evidence_prompt -> 3. generate_response."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    mock_build_prompt.return_value = "Prompt Text"

    # Attach all mocks to a parent Mock manager to record execution order
    manager = Mock()
    manager.attach_mock(mock_retrieve_tickets, "retrieve_tickets")
    manager.attach_mock(mock_build_prompt, "build_evidence_prompt")
    manager.attach_mock(mock_gemini_client.generate_response, "generate_response")

    service = MemoService(gemini_client=mock_gemini_client)
    service.generate_evidence_memo(df, keywords, proposal)

    expected_call_names = ["retrieve_tickets", "build_evidence_prompt", "generate_response"]
    actual_call_names = [call_item[0] for call_item in manager.mock_calls]

    assert actual_call_names == expected_call_names


# 10. Verify the final return value is exactly the Gemini response
@patch("src.memo_service.retrieve_tickets")
@patch("src.memo_service.build_evidence_prompt")
def test_final_return_value_is_gemini_response(
    mock_build_prompt, mock_retrieve_tickets, sample_inputs, mock_gemini_client
):
    """Verify that the return value of generate_evidence_memo is identical to Gemini's output string."""
    df, keywords, proposal = sample_inputs
    mock_retrieve_tickets.return_value = df
    mock_build_prompt.return_value = "Prompt"
    exact_gemini_response = "UNIQUE_MEMO_STRING_RESPONSE_12345"
    mock_gemini_client.generate_response.return_value = exact_gemini_response

    service = MemoService(gemini_client=mock_gemini_client)
    result = service.generate_evidence_memo(df, keywords, proposal)

    assert result is exact_gemini_response
