import pandas as pd
import numpy as np
from src.retrieval import retrieve_tickets
from src.evidence_validation import validate_retrieved_evidence
from src.evidence_scoring import EvidenceScoringEngine
from src.evidence_deduplication import EvidenceDeduplicator
from src.evidence_clustering import EvidenceClusterer
from src.decision_engine import DecisionEngine, DecisionResult
from src.historical_calibration import HistoricalCalibrationEngine
from src.prompt_builder import build_evidence_prompt, build_decision_prompt
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
    Orchestrator that coordinates the Confidence Memo and Decision Memo generation pipelines.
    Coordinates retrieval, deduplication, semantic clustering, validation, deterministic scoring,
    multi-criteria decision evaluation, historical calibration logging, prompt building, and Gemini.
    """

    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        scoring_engine: EvidenceScoringEngine | None = None,
        deduplicator: EvidenceDeduplicator | None = None,
        clusterer: EvidenceClusterer | None = None,
        decision_engine: DecisionEngine | None = None,
        calibration_engine: HistoricalCalibrationEngine | None = None,
    ):
        """
        Initialize the MemoService with all pipeline components.
        """
        self.gemini_client = gemini_client or GeminiClient()
        self.scoring_engine = scoring_engine or EvidenceScoringEngine()
        self.deduplicator = deduplicator or EvidenceDeduplicator()
        self.clusterer = clusterer or EvidenceClusterer()
        self.decision_engine = decision_engine or DecisionEngine()
        self.calibration_engine = calibration_engine or HistoricalCalibrationEngine()

    def generate_evidence_memo(
        self,
        df: pd.DataFrame,
        keywords: list[str],
        proposal: str,
        template_path: str = None,
        validated_tickets: pd.DataFrame = None,
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
            validated_tickets (pd.DataFrame): Optional pre-validated tickets DataFrame.

        Returns:
            str: The raw Markdown response from Gemini (the Evidence Memo).
        """
        if validated_tickets is None:
            # 1. Retrieve matching tickets with similarity scores
            matching_tickets, similarity_scores = retrieve_tickets(df, keywords)

            # 2. Validate evidence relevance — filter out tickets below evidence threshold
            validated_tickets, validated_scores = validate_retrieved_evidence(
                matching_tickets, similarity_scores, return_scores=True
            )
            total_retrieved = len(matching_tickets)
        else:
            validated_scores = None
            total_retrieved = len(validated_tickets)

        # 3. Determine scoring based on validated evidence
        scoring_info = self.scoring_engine.calculate_score(
            validated_tickets,
            similarity_scores=validated_scores,
            total_retrieved_count=total_retrieved,
        )

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

    def generate_decision_memo(
        self,
        df: pd.DataFrame,
        keywords: list[str],
        proposal: str,
        engineering_effort: str | int | float = "Medium",
        business_impact: str | int | float = "Medium",
        strategic_alignment: str | int | float = "Medium",
        cost: str | int | float = "Low",
        risk: str | int | float = "Low",
        template_path: str = None,
        record_calibration: bool = True,
        return_decision_result: bool = False,
        validated_tickets: pd.DataFrame = None,
    ) -> str | tuple[str, DecisionResult]:
        """
        Generate a comprehensive Decision Memo that synthesizes customer evidence with
        strategic criteria (effort, impact, alignment, cost, risk) through a deterministic
        decision engine.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame.
            keywords (list[str]): Keywords to retrieve candidate tickets.
            proposal (str): The proposed feature to evaluate.
            engineering_effort: Engineering complexity ('Low' to 'Very High' or 1-5).
            business_impact: Business impact ('Low' to 'Critical' or 1-5).
            strategic_alignment: Strategic alignment ('Low' to 'High' or 1-5).
            cost: Cost scale ('Low' to 'High' or 1-5).
            risk: Risk scale ('Low' to 'High' or 1-5).
            template_path (str): Optional path to custom decision prompt template.
            record_calibration (bool): Whether to record this prediction in the calibration store.
            return_decision_result (bool): If True, returns tuple (memo_str, decision_result).
            validated_tickets (pd.DataFrame): Optional pre-validated tickets.

        Returns:
            str or tuple[str, DecisionResult]: Decision Memo Markdown string (and optional DecisionResult).
        """
        # Step 1: Retrieval & Validation
        if validated_tickets is None:
            matching_tickets, similarity_scores = retrieve_tickets(df, keywords)
            val_tickets, val_scores = validate_retrieved_evidence(
                matching_tickets, similarity_scores, return_scores=True
            )
            total_retrieved = len(matching_tickets)
        else:
            val_tickets = validated_tickets.copy()
            val_scores = None
            total_retrieved = len(val_tickets)

        # Step 2: Evidence Deduplication (Exact + Semantic)
        dedup_tickets, dedup_scores, dedup_stats = self.deduplicator.deduplicate(
            val_tickets, similarity_scores=val_scores
        )

        # Step 3: Evidence Semantic Clustering & Theme Extraction
        clustered_tickets, cluster_stats = self.clusterer.cluster_tickets(
            dedup_tickets
        )

        # Step 4: Deterministic Evidence Scoring
        scoring_info = self.scoring_engine.calculate_score(
            clustered_tickets,
            similarity_scores=dedup_scores,
            total_retrieved_count=total_retrieved,
        )

        # Step 5: Deterministic Decision Engine
        decision_result = self.decision_engine.evaluate_decision(
            evidence_score=scoring_info.get("score", 0),
            engineering_effort=engineering_effort,
            business_impact=business_impact,
            strategic_alignment=strategic_alignment,
            cost=cost,
            risk=risk,
            evidence_summary=scoring_info.get("evidence_summary"),
            deduplication_stats=dedup_stats,
            clustering_stats=cluster_stats,
        )

        # Step 6: Record in Historical Calibration Store
        if record_calibration:
            self.calibration_engine.record_prediction(
                proposal=proposal,
                predicted_score=decision_result.evidence_score,
                predicted_confidence=scoring_info.get("confidence", "Low"),
                predicted_recommendation=decision_result.recommendation,
                metadata={
                    "priority_score": decision_result.priority_score,
                    "decision_tier": decision_result.decision_tier,
                    "business_impact": business_impact,
                    "engineering_effort": engineering_effort,
                },
            )

        # Step 7: Build Decision Prompt for Gemini
        prompt = build_decision_prompt(
            proposal=proposal,
            df_tickets=clustered_tickets,
            decision_info=decision_result.to_dict(),
            scoring_info=scoring_info,
            deduplication_stats=dedup_stats,
            clustering_stats=cluster_stats,
            template_path=template_path,
        )

        # Step 8: Generate Memo via Gemini Client
        memo = self.gemini_client.generate_response(prompt)

        if return_decision_result:
            return memo, decision_result
        return memo
