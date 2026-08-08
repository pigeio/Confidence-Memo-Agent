import pandas as pd
import numpy as np


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

    def calculate_score(self, df: pd.DataFrame) -> dict:
        """
        Calculate a deterministic Evidence Score and Confidence Level for a ticket DataFrame.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame.

        Returns:
            dict: Structured dictionary containing score, confidence, and factor breakdown:
                {
                    "score": int,  # 0 to 100
                    "confidence": str,  # "Low", "Moderate", or "High"
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

        if df.empty:
            return {
                "score": 0,
                "confidence": "Low",
                "factors": {
                    "ticket_volume": 0.0,
                    "severity": 0.0,
                    "sentiment_consistency": 0.0,
                    "recency": 0.0,
                    "diversity": 0.0,
                },
            }

        # Calculate individual factors
        volume_score = self._calculate_ticket_volume(df)
        sentiment_score = self._calculate_sentiment_consistency(df)
        severity_score = self._calculate_severity(df)
        recency_score = self._calculate_recency(df)
        diversity_score = self._calculate_diversity(df)

        # Sum raw factor points
        raw_total = (
            volume_score + sentiment_score + severity_score + recency_score + diversity_score
        )

        # Clamp total score between 0 and 100
        score = int(np.clip(round(raw_total), 0, 100))

        # Map score to confidence level
        confidence = self._map_confidence_level(score)

        return {
            "score": score,
            "confidence": confidence,
            "factors": {
                "ticket_volume": round(volume_score, 1),
                "severity": round(severity_score, 1),
                "sentiment_consistency": round(sentiment_score, 1),
                "recency": round(recency_score, 1),
                "diversity": round(diversity_score, 1),
            },
        }

    def _calculate_ticket_volume(self, df: pd.DataFrame) -> float:
        """
        Calculate ticket volume score (Max 30 pts).
        Scale: 1 ticket = 6 pts, up to 5+ tickets = 30 pts max.
        """
        count = len(df)
        return min(30.0, count * 6.0)

    def _calculate_sentiment_consistency(self, df: pd.DataFrame) -> float:
        """
        Calculate sentiment consistency score based on message detail and clarity (Max 25 pts).
        """
        if df.empty or "message" not in df.columns:
            return 10.0

        messages = df["message"].astype(str).str.strip()
        avg_len = messages.str.len().mean()

        if avg_len >= 30:
            return 25.0
        elif avg_len >= 15:
            return 18.0
        else:
            return 10.0

    def _calculate_severity(self, df: pd.DataFrame) -> float:
        """
        Calculate severity and urgency score based on pain/urgency keywords (Max 20 pts).
        """
        if df.empty or "message" not in df.columns:
            return 0.0

        messages = df["message"].astype(str).str.lower()
        severity_count = 0

        for msg in messages:
            if any(keyword in msg for keyword in self.SEVERITY_KEYWORDS):
                severity_count += 1

        # 4 pts per severe ticket up to 20 pts
        return min(20.0, severity_count * 4.0)

    def _calculate_recency(self, df: pd.DataFrame) -> float:
        """
        Calculate recency score (Max 15 pts).
        If created_at is available, evaluates ticket timestamps. Default baseline: 10 pts.
        """
        if "created_at" in df.columns and not df["created_at"].isna().all():
            try:
                dates = pd.to_datetime(df["created_at"], errors="coerce")
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    # Give higher score if tickets are recent
                    return 15.0
            except Exception:
                pass
        return 10.0

    def _calculate_diversity(self, df: pd.DataFrame) -> float:
        """
        Calculate user/ticket diversity score (Max 10 pts).
        Rewards non-duplicate, unique ticket entries.
        """
        total = len(df)
        if total == 0:
            return 0.0

        if "ticket_id" in df.columns:
            unique_count = df["ticket_id"].nunique()
        else:
            unique_count = total

        ratio = unique_count / total
        return round(ratio * 10.0, 1)

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
