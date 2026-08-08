import numpy as np
import pandas as pd
from src.config import EVIDENCE_SIMILARITY_THRESHOLD


def validate_retrieved_evidence(
    tickets: pd.DataFrame,
    similarity_scores: np.ndarray,
    threshold: float = EVIDENCE_SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter retrieved tickets to retain only those with similarity scores
    at or above the evidence threshold.

    This is a pure filtering function with no side effects. It never
    calculates confidence, builds prompts, or calls Gemini.

    Parameters:
        tickets (pd.DataFrame): Retrieved support tickets DataFrame.
        similarity_scores (np.ndarray): 1D array of cosine similarity scores,
            one per ticket row, in the same order as the DataFrame.
        threshold (float): Minimum similarity score to qualify as relevant evidence.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only tickets whose similarity
            scores meet or exceed the threshold. Preserves original ranking order.
            Returns an empty DataFrame (with original columns) if nothing passes.

    Raises:
        TypeError: If tickets is not a DataFrame or similarity_scores is not an ndarray.
        ValueError: If the length of similarity_scores does not match the number of ticket rows.
    """
    if not isinstance(tickets, pd.DataFrame):
        raise TypeError("tickets must be a pandas DataFrame")

    if not isinstance(similarity_scores, np.ndarray):
        raise TypeError("similarity_scores must be a numpy ndarray")

    if tickets.empty:
        return tickets.copy()

    if len(similarity_scores) != len(tickets):
        raise ValueError(
            f"Length mismatch: tickets has {len(tickets)} rows but "
            f"similarity_scores has {len(similarity_scores)} elements"
        )

    # Build boolean mask for scores at or above the threshold
    mask = similarity_scores >= threshold

    # Apply mask preserving original row ordering
    validated = tickets[mask].copy()

    return validated
