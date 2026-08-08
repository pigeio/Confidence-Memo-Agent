import numpy as np
from src.config import DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD


class SimilarityEngine:
    """
    Pure NumPy vector engine for computing cosine similarity and selecting top-k candidates.
    Assumes embeddings are pre-normalized to unit length (L2 norm = 1.0).
    """

    @staticmethod
    def compute_cosine_similarity(
        ticket_embeddings: np.ndarray, query_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Compute dot product cosine similarity between pre-normalized ticket embeddings and query embedding.

        Parameters:
            ticket_embeddings (np.ndarray): 2D array of shape (N, D).
            query_embedding (np.ndarray): 1D or 2D array of shape (D,) or (1, D).

        Returns:
            np.ndarray: 1D array of shape (N,) containing similarity scores.
        """
        if not isinstance(ticket_embeddings, np.ndarray) or not isinstance(
            query_embedding, np.ndarray
        ):
            raise TypeError("Embeddings must be numpy ndarrays")

        if ticket_embeddings.size == 0:
            return np.array([], dtype=np.float32)

        if ticket_embeddings.ndim != 2:
            raise ValueError(
                f"ticket_embeddings must be a 2D array, got {ticket_embeddings.ndim}D"
            )

        # Flatten query if 2D
        q_vec = query_embedding.flatten()
        if ticket_embeddings.shape[1] != q_vec.shape[0]:
            raise ValueError(
                f"Dimension mismatch: ticket embeddings dim ({ticket_embeddings.shape[1]}) "
                f"does not match query vector dim ({q_vec.shape[0]})"
            )

        # Compute dot product (equals cosine similarity for L2-normalized vectors)
        scores = np.dot(ticket_embeddings, q_vec)
        return scores

    @staticmethod
    def rank_and_filter(
        scores: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> list[tuple[int, float]]:
        """
        Rank indices by score descending and filter by similarity threshold up to top_k.

        Parameters:
            scores (np.ndarray): 1D array of similarity scores.
            top_k (int): Maximum number of top candidates to return.
            threshold (float): Minimum similarity threshold score.

        Returns:
            list[tuple[int, float]]: List of (index, similarity_score) tuples sorted descending.
        """
        if not isinstance(scores, np.ndarray):
            raise TypeError("scores must be a numpy ndarray")

        if scores.size == 0 or top_k <= 0:
            return []

        # Sort indices descending by score
        sorted_indices = np.argsort(scores)[::-1]

        ranked_results = []
        for idx in sorted_indices:
            score = float(scores[idx])
            if score >= threshold:
                ranked_results.append((int(idx), score))
                if len(ranked_results) == top_k:
                    break

        return ranked_results
