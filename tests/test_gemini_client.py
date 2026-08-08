import unittest
from unittest.mock import MagicMock, patch
from google.genai import errors
from src.gemini_client import GeminiClient


class TestGeminiClient(unittest.TestCase):
    @patch("src.gemini_client.genai.Client")
    def test_successful_response(self, mock_genai_client_class):
        # Setup mock client and response
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "This is a confidence memo."
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="fake-key")
        response = client.generate_response("Test prompt")

        self.assertEqual(response, "This is a confidence memo.")
        mock_client.models.generate_content.assert_called_once_with(
            model="gemini-1.5-flash", contents="Test prompt"
        )

    def test_empty_prompt_validation(self):
        client = GeminiClient(api_key="fake-key")
        with self.assertRaises(ValueError):
            client.generate_response("")
        with self.assertRaises(ValueError):
            client.generate_response("   ")

    @patch("src.gemini_client.genai.Client")
    def test_empty_response_text(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = ""  # Empty text
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient(api_key="fake-key")
        with self.assertRaises(ValueError):
            client.generate_response("Test prompt")

    @patch("src.gemini_client.genai.Client")
    def test_transient_error_retry_success(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Simulate a transient 429 rate limit error on the first call, then success
        mock_error = errors.APIError(message="Rate limit exceeded", code=429)
        mock_response = MagicMock()
        mock_response.text = "Success after retry."
        mock_client.models.generate_content.side_effect = [mock_error, mock_response]

        # Use initial_delay = 0.01 to speed up test execution
        client = GeminiClient(api_key="fake-key", max_retries=2, initial_delay=0.01)
        response = client.generate_response("Test prompt")

        self.assertEqual(response, "Success after retry.")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

    @patch("src.gemini_client.genai.Client")
    def test_non_transient_error_no_retry(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Simulate a non-transient 403 Forbidden error
        mock_error = errors.APIError(message="Invalid API Key", code=403)
        mock_client.models.generate_content.side_effect = mock_error

        client = GeminiClient(api_key="fake-key", max_retries=2, initial_delay=0.01)

        with self.assertRaises(errors.APIError):
            client.generate_response("Test prompt")

        # Should only be called once, because 403 is not retryable
        mock_client.models.generate_content.assert_called_once()

    @patch("src.gemini_client.genai.Client")
    def test_retry_exhaustion(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Simulate persistent 503 Service Unavailable errors
        mock_error = errors.APIError(message="Service Unavailable", code=503)
        mock_client.models.generate_content.side_effect = mock_error

        client = GeminiClient(api_key="fake-key", max_retries=2, initial_delay=0.01)

        with self.assertRaises(errors.APIError):
            client.generate_response("Test prompt")

        # 1 initial try + 2 retries = 3 calls total
        self.assertEqual(mock_client.models.generate_content.call_count, 3)

    @patch("src.gemini_client.genai.Client")
    def test_general_exception_retry_success(self, mock_genai_client_class):
        mock_client = MagicMock()
        mock_genai_client_class.return_value = mock_client

        # Simulate a general ConnectionError then success
        mock_error = ConnectionError("Connection reset by peer")
        mock_response = MagicMock()
        mock_response.text = "Recovered from net issues."
        mock_client.models.generate_content.side_effect = [mock_error, mock_response]

        client = GeminiClient(api_key="fake-key", max_retries=1, initial_delay=0.01)
        response = client.generate_response("Test prompt")

        self.assertEqual(response, "Recovered from net issues.")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)


if __name__ == "__main__":
    unittest.main()
