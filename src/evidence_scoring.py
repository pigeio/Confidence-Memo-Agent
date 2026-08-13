import pandas as pd
import numpy as np
from src.config import (
    SINGLE_TICKET_SCORE_CAP,
    TWO_TICKET_SCORE_CAP,
    MIN_TICKETS_FOR_HIGH_CONFIDENCE,
)


class EvidenceScoringEngine:
    """
    Deterministic scoring engine that evaluates support ticket evidence
    to calculate a numerical Evidence Score (0-100) and Confidence Level (Low/Moderate/High).

    The confidence score is computed using five weighted factors:
    - Ticket Volume (Max 30 pts)
    - Sentiment Consistency (Max 25 pts)
    - Severity & Urgency (Max 20 pts)
    - Recency (Max 15 pts)
    - User Diversity (Max 10 pts)
    """

    SEVERITY_KEYWORDS = {
        "crash", "crashes", "hurts", "hurt", "burning", "blinding",
        "cannot", "can't", "broken", "severe", "freeze", "fail",
        "urgent", "error", "glare", "pain", "issue", "bug"
    }

    def calculate_score(
        self,
        df: pd.DataFrame,
        similarity_scores: np.ndarray = None,
        total_retrieved_count: int = None,
    ) -> dict:
        """
        Calculate a deterministic Evidence Score and Confidence Level for a ticket DataFrame.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame.
            similarity_scores (np.ndarray): Optional 1D numpy array of cosine similarity scores.
            total_retrieved_count (int): Optional count of candidate tickets before validation.

        Returns:
            dict: Structured dictionary containing score, confidence, evidence_summary, and factor breakdown:
                {
                    "score": int,  # 0 to 100
                    "confidence": str,  # "Low", "Moderate", or "High"
                    "evidence_summary": {
                        "validated_tickets": int,
                        "retrieved_tickets": int,
                        "rejected_tickets": int,
                        "average_similarity": float,
                        "sample_size": str  # "Zero", "Small", "Medium", or "Large"
                    },
                    "factors": {
                        "ticket_volume": float,
                        "severity": float,
                        "sentiment_consistency": float,
                        "recency": float,
                        "diversity": float
                    }
                }

        Raises:
            TypeError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        retrieved_count = total_retrieved_count if total_retrieved_count is not None else len(df)
        rejected_count = max(0, retrieved_count - len(df))

        if df.empty:
            return {
                "score": 0,
                "confidence": "Low",
                "evidence_summary": {
                    "validated_tickets": 0,
                    "retrieved_tickets": retrieved_count,
                    "rejected_tickets": rejected_count,
                    "average_similarity": 0.0,
                    "sample_size": "Zero",
                },
                "factors": {
                    "ticket_volume": 0.0,
                    "severity": 0.0,
                    "sentiment_consistency": 0.0,
                    "recency": 0.0,
                    "diversity": 0.0,
                },
            }

        # Format and validate similarity scores array
        if similarity_scores is not None and len(similarity_scores) == len(df):
            sims = np.clip(np.array(similarity_scores, dtype=np.float32), 0.0, 1.0)
        else:
            sims = np.ones(len(df), dtype=np.float32)

        mean_sim = float(np.mean(sims)) if len(sims) > 0 else 1.0

        # Calculate individual factors weighted by similarity
        volume_score = self._calculate_ticket_volume(df, sims)
        sentiment_score = self._calculate_sentiment_consistency(df, mean_sim)
        severity_score = self._calculate_severity(df, sims)
        recency_score = self._calculate_recency(df, mean_sim)
        diversity_score = self._calculate_diversity(df, mean_sim)

        # Sum raw factor points
        raw_total = (
            volume_score + sentiment_score + severity_score + recency_score + diversity_score
        )

        # Apply small dataset confidence score caps
        count = len(df)
        if count == 1:
            score_cap = SINGLE_TICKET_SCORE_CAP  # 39 max (Low)
            sample_size_label = "Small"
        elif count == 2:
            score_cap = TWO_TICKET_SCORE_CAP  # 69 max (Moderate)
            sample_size_label = "Small"
        elif count < 6:
            score_cap = 100
            sample_size_label = "Medium"
        else:
            score_cap = 100
            sample_size_label = "Large"

        # Clamp total score between 0 and score_cap
        score = int(np.clip(round(raw_total), 0, score_cap))

        # Map score to confidence level
        confidence = self._map_confidence_level(score)

        evidence_summary = {
            "validated_tickets": count,
            "retrieved_tickets": retrieved_count,
            "rejected_tickets": rejected_count,
            "average_similarity": round(mean_sim, 2),
            "sample_size": sample_size_label,
        }

        return {
            "score": score,
            "confidence": confidence,
            "evidence_summary": evidence_summary,
            "factors": {
                "ticket_volume": round(volume_score, 1),
                "severity": round(severity_score, 1),
                "sentiment_consistency": round(sentiment_score, 1),
                "recency": round(recency_score, 1),
                "diversity": round(diversity_score, 1),
            },
        }

    def _calculate_ticket_volume(self, df: pd.DataFrame, sims: np.ndarray) -> float:
        """
        Calculate ticket volume score weighted by similarity scores (Max 30 pts).
        Scale: 1 ticket = 6 pts weighted by similarity, up to 30 pts max.
        """
        weighted_count = float(np.sum(sims))
        return min(30.0, weighted_count * 6.0)

    def _calculate_sentiment_consistency(self, df: pd.DataFrame, mean_sim: float) -> float:
        """
        Calculate sentiment consistency score based on message detail and clarity (Max 25 pts).
        Weighted by average similarity.
        """
        if df.empty or "message" not in df.columns:
            return 10.0 * mean_sim

        messages = df["message"].astype(str).str.strip()
        avg_len = messages.str.len().mean()

        if avg_len >= 30:
            base = 25.0
        elif avg_len >= 15:
            base = 18.0
        else:
            base = 10.0

        return base * mean_sim

    def _calculate_severity(self, df: pd.DataFrame, sims: np.ndarray) -> float:
        """
        Calculate severity and urgency score based on pain/urgency keywords (Max 20 pts).
        Weighted by ticket similarity scores.
        """
        if df.empty or "message" not in df.columns:
            return 0.0

        messages = df["message"].astype(str).str.lower()
        weighted_severity = 0.0

        for idx, msg in enumerate(messages):
            if any(keyword in msg for keyword in self.SEVERITY_KEYWORDS):
                sim = sims[idx] if idx < len(sims) else 1.0
                weighted_severity += 4.0 * sim

        return min(20.0, weighted_severity)

    def _calculate_recency(self, df: pd.DataFrame, mean_sim: float) -> float:
        """
        Calculate recency score (Max 15 pts).
        If created_at is available, evaluates ticket timestamps. Default baseline: 10 pts.
        Weighted by average similarity.
        """
        base = 10.0
        if "created_at" in df.columns and not df["created_at"].isna().all():
            try:
                dates = pd.to_datetime(df["created_at"], errors="coerce")
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    base = 15.0
            except Exception:
                pass
        return base * mean_sim

    def _calculate_diversity(self, df: pd.DataFrame, mean_sim: float) -> float:
        """
        Calculate user/ticket diversity score (Max 10 pts).
        Rewards non-duplicate, unique ticket entries, weighted by mean similarity.
        """
        total = len(df)
        if total == 0:
            return 0.0

        if "ticket_id" in df.columns:
            unique_count = df["ticket_id"].nunique()
        else:
            unique_count = total

        ratio = unique_count / total
        return round(ratio * 10.0 * mean_sim, 1)

    def _map_confidence_level(self, score: int) -> str:
        """
        Map numerical Evidence Score (0-100) to Confidence Level.
        0-39   -> Low
        40-69  -> Moderate
        70-100 -> High
        """
        if score < 40:
            return "Low"
        elif score < 70:
            return "Moderate"
        else:
            return "High"

