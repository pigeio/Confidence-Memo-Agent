import pytest
from unittest.mock import MagicMock, patch, call
from google.genai import errors
from src.gemini_client import GeminiClient


@pytest.fixture
def mock_genai_client():
    """
    Pytest fixture that patches google.genai.Client and returns a mock client instance.
    """
    with patch("src.gemini_client.genai.Client") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def client(mock_genai_client):
    """
    Pytest fixture providing a GeminiClient initialized with a fake API key
    and zero initial delay to ensure deterministic fast tests.
    """
    return GeminiClient(api_key="fake-api-key", max_retries=3, initial_delay=1.0)


# 1. Successful API response
def test_successful_api_response(client, mock_genai_client):
    """Verify that generate_response returns text output on a successful API response."""
    mock_response = MagicMock()
    mock_response.text = "Analysis Memo Output"
    mock_genai_client.models.generate_content.return_value = mock_response

    result = client.generate_response("Analyze tickets proposal")

    assert result == "Analysis Memo Output"
    mock_genai_client.models.generate_content.assert_called_once_with(
        model="gemini-3.5-flash",
        contents="Analyze tickets proposal"
    )


# 2. Retry on HTTP 429 (rate limit)
@patch("src.gemini_client.time.sleep")
def test_retry_on_http_429_rate_limit(mock_sleep, client, mock_genai_client):
    """Verify transient HTTP 429 rate limit errors trigger retry and succeed on retry."""
    rate_limit_error = errors.APIError(429, {"message": "Rate limit exceeded"})
    successful_response = MagicMock()
    successful_response.text = "Success after rate limit retry"

    mock_genai_client.models.generate_content.side_effect = [rate_limit_error, successful_response]

    result = client.generate_response("Valid prompt")

    assert result == "Success after rate limit retry"
    assert mock_genai_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


# 3. Retry on HTTP 503 (temporary server error)
@patch("src.gemini_client.time.sleep")
def test_retry_on_http_503_server_error(mock_sleep, client, mock_genai_client):
    """Verify transient HTTP 503 Service Unavailable errors trigger retry."""
    server_error = errors.APIError(503, {"message": "Service Unavailable"})
    successful_response = MagicMock()
    successful_response.text = "Success after server error retry"

    mock_genai_client.models.generate_content.side_effect = [server_error, successful_response]

    result = client.generate_response("Valid prompt")

    assert result == "Success after server error retry"
    assert mock_genai_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


# 4. Retry on network timeout
@patch("src.gemini_client.time.sleep")
def test_retry_on_network_timeout(mock_sleep, client, mock_genai_client):
    """Verify network timeouts / connection errors trigger retry logic."""
    timeout_error = TimeoutError("Request timed out")
    successful_response = MagicMock()
    successful_response.text = "Success after timeout"

    mock_genai_client.models.generate_content.side_effect = [timeout_error, successful_response]

    result = client.generate_response("Valid prompt")

    assert result == "Success after timeout"
    assert mock_genai_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


# 5. Stop retrying after configured maximum retries
@patch("src.gemini_client.time.sleep")
def test_stop_retrying_after_max_retries(mock_sleep, mock_genai_client):
    """Verify API stops retrying once max_retries limit is reached."""
    custom_client = GeminiClient(api_key="fake-key", max_retries=2, initial_delay=1.0)
    server_error = errors.APIError(503, {"message": "Persistent 503"})

    mock_genai_client.models.generate_content.side_effect = [
        server_error,
        server_error,
        server_error
    ]

    with pytest.raises(errors.APIError):
        custom_client.generate_response("Valid prompt")

    # 1 initial try + 2 retries = 3 total calls
    assert mock_genai_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


# 6. Invalid API key (HTTP 403) should fail immediately
@patch("src.gemini_client.time.sleep")
def test_invalid_api_key_fails_immediately(mock_sleep, client, mock_genai_client):
    """Verify non-transient HTTP 403 Forbidden errors fail immediately without retrying."""
    forbidden_error = errors.APIError(403, {"message": "Invalid API Key"})
    mock_genai_client.models.generate_content.side_effect = forbidden_error

    with pytest.raises(errors.APIError) as exc_info:
        client.generate_response("Valid prompt")

    assert exc_info.value.code == 403
    mock_genai_client.models.generate_content.assert_called_once()
    mock_sleep.assert_not_called()


# 7. Invalid request should fail immediately
@patch("src.gemini_client.time.sleep")
def test_invalid_request_fails_immediately(mock_sleep, client, mock_genai_client):
    """Verify invalid input prompts (empty/whitespace) or HTTP 400 Bad Request fail immediately."""
    # Empty prompt input validation
    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        client.generate_response("")

    with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
        client.generate_response("   ")

    mock_genai_client.models.generate_content.assert_not_called()

    # HTTP 400 Bad Request from API
    bad_request_error = errors.APIError(400, {"message": "Bad Request parameters"})
    mock_genai_client.models.generate_content.side_effect = bad_request_error

    with pytest.raises(errors.APIError):
        client.generate_response("Valid string prompt")

    mock_genai_client.models.generate_content.assert_called_once()
    mock_sleep.assert_not_called()


# 8. Exceptions are propagated correctly
def test_exceptions_propagated_correctly(client, mock_genai_client):
    """Verify underlying APIError and custom exceptions are raised with original context."""
    custom_exception = RuntimeError("Custom client failure")
    mock_genai_client.models.generate_content.side_effect = custom_exception

    # Client configured with 0 retries to test immediate propagation
    no_retry_client = GeminiClient(api_key="fake-key", max_retries=0)

    with pytest.raises(RuntimeError) as exc_info:
        no_retry_client.generate_response("Valid prompt")

    assert exc_info.value is custom_exception


# 9. Retry backoff logic is invoked correctly (exponential delay)
@patch("src.gemini_client.time.sleep")
def test_exponential_backoff_sleep_delays(mock_sleep, mock_genai_client):
    """Verify time.sleep is called with exponentially increasing delays (initial_delay * 2^n)."""
    client_backoff = GeminiClient(api_key="fake-key", max_retries=3, initial_delay=2.0)
    server_error = errors.APIError(500, {"message": "Internal Server Error"})

    mock_genai_client.models.generate_content.side_effect = [
        server_error,
        server_error,
        server_error,
        server_error
    ]

    with pytest.raises(errors.APIError):
        client_backoff.generate_response("Valid prompt")

    # Expected exponential sleep calls: 2.0s, 4.0s, 8.0s
    expected_calls = [call(2.0), call(4.0), call(8.0)]
    mock_sleep.assert_has_calls(expected_calls, any_order=False)
    assert mock_sleep.call_count == 3
