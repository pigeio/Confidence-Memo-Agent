import pandas as pd
from src.retrieval import retrieve_tickets
from src.evidence_validation import validate_retrieved_evidence
from src.evidence_scoring import EvidenceScoringEngine
from src.prompt_builder import build_evidence_prompt
from src.gemini_client import GeminiClient


# Pre-built scoring result for the "no relevant evidence" short-circuit
_NO_EVIDENCE_SCORING = {
    "score": 0,
    "confidence": "Low",
    "reason": "No relevant customer evidence was found for this proposal.",
    "factors": {
        "ticket_volume": 0.0,
        "severity": 0.0,
        "sentiment_consistency": 0.0,
        "recency": 0.0,
        "diversity": 0.0,
    },
}


class MemoService:
    """
    Orchestrator that coordinates the Confidence Memo generation pipeline.
    It retrieves relevant tickets, validates evidence relevance, calculates a
    deterministic Evidence Score, constructs an analyst prompt, and queries Gemini.
    """

    def __init__(
        self,
        gemini_client: GeminiClient = None,
        scoring_engine: EvidenceScoringEngine = None,
    ):
        """
        Initialize the MemoService.

        Parameters:
            gemini_client (GeminiClient): An instance of GeminiClient. If not provided,
                                         it will initialize a default client.
            scoring_engine (EvidenceScoringEngine): An instance of EvidenceScoringEngine.
                                                   If not provided, initializes default engine.
        """
        self.gemini_client = gemini_client or GeminiClient()
        self.scoring_engine = scoring_engine or EvidenceScoringEngine()

    def generate_evidence_memo(
        self,
        df: pd.DataFrame,
        keywords: list[str],
        proposal: str,
        template_path: str = None,
    ) -> str:
        """
        Generate an Evidence Memo for a given feature proposal by retrieving tickets,
        validating evidence relevance, calculating deterministic evidence scores,
        building the prompt, and querying Gemini.

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
        # 1. Retrieve matching tickets with similarity scores
        matching_tickets, similarity_scores = retrieve_tickets(df, keywords)

        # 2. Validate evidence relevance — filter out tickets below evidence threshold
        validated_tickets = validate_retrieved_evidence(
            matching_tickets, similarity_scores
        )

        # 3. Determine scoring based on validated evidence
        if validated_tickets.empty:
            # Short-circuit: no relevant evidence found
            scoring_info = _NO_EVIDENCE_SCORING.copy()
        else:
            scoring_info = self.scoring_engine.calculate_score(validated_tickets)

        # 4. Build the prompt including the pre-calculated scoring summary
        prompt = build_evidence_prompt(
            proposal,
            validated_tickets,
            scoring_info=scoring_info,
            template_path=template_path,
        )

        # 5. Generate response using the Gemini client
        memo = self.gemini_client.generate_response(prompt)

        return memo
