import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.memo_service import MemoService


class TestMemoService(unittest.TestCase):
    @patch("src.memo_service.retrieve_tickets")
    @patch("src.memo_service.build_evidence_prompt")
    def test_successful_orchestration(self, mock_build_prompt, mock_retrieve_tickets):
        # 1. Setup inputs
        df_input = pd.DataFrame({"topic": ["A"], "message": ["B"]})
        keywords = ["test"]
        proposal = "Implement test feature"

        # 2. Setup mock return values
        df_retrieved = pd.DataFrame({"topic": ["A matched"], "message": ["B matched"]})
        mock_retrieve_tickets.return_value = df_retrieved
        mock_build_prompt.return_value = "Mocked Prompt Content"

        mock_gemini_client = MagicMock()
        mock_gemini_client.generate_response.return_value = "Mocked Evidence Memo Output"

        # 3. Initialize service and run
        service = MemoService(gemini_client=mock_gemini_client)
        result = service.generate_evidence_memo(
            df_input, keywords, proposal, template_path="custom_path.txt"
        )

        # 4. Asserts
        self.assertEqual(result, "Mocked Evidence Memo Output")
        mock_retrieve_tickets.assert_called_once_with(df_input, keywords)
        mock_build_prompt.assert_called_once_with(
            proposal, df_retrieved, template_path="custom_path.txt"
        )
        mock_gemini_client.generate_response.assert_called_once_with(
            "Mocked Prompt Content"
        )


if __name__ == "__main__":
    unittest.main()
