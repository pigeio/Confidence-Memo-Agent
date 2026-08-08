import pandas as pd
from src.retrieval import retrieve_tickets
from src.prompt_builder import build_evidence_prompt
from src.gemini_client import GeminiClient


class MemoService:
    """
    Orchestrator that coordinates the Confidence Memo generation pipeline.
    It retrieves relevant tickets, constructs a calibrated prompt, and queries Gemini.
    """

    def __init__(self, gemini_client: GeminiClient = None):
        """
        Initialize the MemoService.

        Parameters:
            gemini_client (GeminiClient): An instance of GeminiClient. If not provided,
                                         it will initialize a default client.
        """
        self.gemini_client = gemini_client or GeminiClient()

    def generate_evidence_memo(
        self,
        df: pd.DataFrame,
        keywords: list[str],
        proposal: str,
        template_path: str = None,
    ) -> str:
        """
        Generate an Evidence Memo for a given feature proposal by querying Gemini
        with support tickets matched by the given keywords.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame.
            keywords (list[str]): Keywords to retrieve relevant tickets.
            proposal (str): The proposed feature to analyze.
            template_path (str): Optional path to a custom prompt template file.

        Returns:
            str: The raw Markdown response from Gemini (the Evidence Memo).

        Raises:
            TypeError: If inputs are invalid types.
            ValueError: If inputs fail validation constraints (e.g. empty keywords/proposal).
        """
        # 1. Retrieve matching tickets (validates df and keywords internally)
        matching_tickets = retrieve_tickets(df, keywords)

        # 2. Build the prompt (validates proposal and df columns internally)
        prompt = build_evidence_prompt(
            proposal, matching_tickets, template_path=template_path
        )

        # 3. Generate response using the Gemini client
        memo = self.gemini_client.generate_response(prompt)

        return memo
