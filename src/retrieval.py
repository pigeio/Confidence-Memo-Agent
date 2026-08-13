import numpy as np
import pandas as pd
from src.semantic_search import search_tickets


def retrieve_tickets(
    df: pd.DataFrame,
    keywords: list[str] | str,
    top_k: int = 5,
    threshold: float = 0.2,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Retrieve support tickets that semantically match any of the given keywords
    or search query in the topic or message columns.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the support tickets.
                           Must have 'topic' and 'message' columns.
        keywords (list[str] | str): A list of keywords or query string to search for.
        top_k (int): Maximum number of tickets to retrieve (default 5).
        threshold (float): Minimum cosine similarity threshold (default 0.2).

    Returns:
        tuple[pd.DataFrame, np.ndarray]: A tuple of:
            - DataFrame containing only the matching tickets.
            - 1D numpy array of similarity scores for each matched ticket.
    """
    return search_tickets(df, keywords, top_k=top_k, threshold=threshold)