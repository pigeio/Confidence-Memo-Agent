import numpy as np
import pandas as pd
from src.config import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD
from src.embedding_generator import EmbeddingGenerator
from src.similarity_engine import SimilarityEngine


class SemanticSearchEngine:
    """
    Search coordinator combining EmbeddingGenerator and SimilarityEngine.
    Encodes support tickets and queries into vectors and calculates cosine similarity.
    """

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator = None,
        similarity_engine: SimilarityEngine = None,
    ):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.similarity_engine = similarity_engine or SimilarityEngine()

    def search(
        self,
        df: pd.DataFrame,
        query: str | list[str],
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Perform semantic search over ticket DataFrame using query text.

        Parameters:
            df (pd.DataFrame): DataFrame of support tickets.
            query (str | list[str]): Query string or list of keywords.
            top_k (int): Maximum number of tickets to retrieve.
            threshold (float): Minimum cosine similarity score threshold.

        Returns:
            tuple[pd.DataFrame, np.ndarray]: A tuple of:
                - Subset DataFrame containing top-k relevant tickets sorted by relevance.
                - 1D numpy array of similarity scores corresponding to each returned ticket.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            return df.copy(), np.array([], dtype=np.float32)

        # Format query string
        if isinstance(query, list):
            if not query:
                raise ValueError("query list cannot be empty")
            valid_items = [q.strip() for q in query if isinstance(q, str) and q.strip()]
            if not valid_items:
                raise ValueError("query list contains no valid non-empty strings")
            query_str = " ".join(valid_items)
        elif isinstance(query, str):
            query_str = query.strip()
        else:
            raise TypeError("query must be a string or a list of strings")

        if not query_str:
            raise ValueError("query cannot be empty or whitespace-only")

        # Encode DataFrame and query
        ticket_embeddings = self.embedding_generator.encode_dataframe(df)
        query_embedding = self.embedding_generator.encode_texts(query_str)

        # Compute cosine similarity matrix
        scores = self.similarity_engine.compute_cosine_similarity(
            ticket_embeddings, query_embedding
        )

        # Rank and filter top candidates
        ranked = self.similarity_engine.rank_and_filter(
            scores, top_k=top_k, threshold=threshold
        )

        if not ranked:
            return pd.DataFrame(columns=df.columns), np.array([], dtype=np.float32)

        indices = [idx for idx, _ in ranked]
        matched_scores = np.array([score for _, score in ranked], dtype=np.float32)
        matched_df = df.iloc[indices].copy()
        return matched_df, matched_scores


# Module-level singleton instance for zero-reinitialization overhead
_search_engine_instance: SemanticSearchEngine | None = None


def search_tickets(
    df: pd.DataFrame,
    query: str | list[str],
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Module-level function wrapper utilizing a singleton SemanticSearchEngine.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: Matched tickets and their similarity scores.
    """
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = SemanticSearchEngine()
    return _search_engine_instance.search(df, query, top_k=top_k, threshold=threshold)
