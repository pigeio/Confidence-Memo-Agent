import logging
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from src.config import CLUSTERING_DISTANCE_THRESHOLD
from src.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)


class EvidenceClusterer:
    """
    Deterministic Semantic Evidence Clustering Engine.
    Groups customer feedback into distinct semantic clusters using dense vector embeddings,
    assigns cluster IDs, identifies central medoid themes, and calculates cluster statistics.
    """

    def __init__(
        self,
        distance_threshold: float = CLUSTERING_DISTANCE_THRESHOLD,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """
        Initialize the EvidenceClusterer.

        Parameters:
            distance_threshold (float): Maximum cosine distance (0.0 to 2.0) for items to be in the same cluster.
            embedding_generator (EmbeddingGenerator): Optional embedding generator instance.
        """
        if not (0.0 <= distance_threshold <= 2.0):
            raise ValueError("distance_threshold must be between 0.0 and 2.0")

        self.distance_threshold = float(distance_threshold)
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def cluster_tickets(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Cluster support tickets semantically and compute detailed cluster statistics.

        Parameters:
            df (pd.DataFrame): Support tickets DataFrame (must contain 'topic' and 'message' columns).
            embeddings (np.ndarray): Optional pre-computed normalized embeddings matrix (N, D).

        Returns:
            tuple: (
                clustered_df (pd.DataFrame): Copy of DataFrame with added 'cluster_id' column,
                stats (dict): Cluster statistics including theme labels and distribution
            )

        Raises:
            TypeError: If df is not a DataFrame or embeddings is not a numpy array.
            ValueError: If df is missing required columns or embeddings shape mismatches.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if embeddings is not None:
            if not isinstance(embeddings, np.ndarray):
                raise TypeError("embeddings must be a numpy ndarray")
            if len(embeddings) != len(df):
                raise ValueError(
                    f"embeddings length ({len(embeddings)}) does not match DataFrame length ({len(df)})"
                )

        total_count = len(df)
        if total_count == 0:
            clustered_df = df.copy()
            clustered_df["cluster_id"] = pd.Series(dtype=int)
            stats = {
                "total_clusters": 0,
                "average_cluster_size": 0.0,
                "cluster_distribution": {},
                "clusters": [],
            }
            return clustered_df, stats

        required_cols = {"topic", "message"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"df is missing required columns: {missing_cols}")

        if total_count == 1:
            clustered_df = df.copy()
            clustered_df["cluster_id"] = 0
            ticket_id = (
                clustered_df.iloc[0].get("ticket_id", 0)
                if "ticket_id" in clustered_df.columns
                else 0
            )
            topic = clustered_df.iloc[0].get("topic", "N/A")
            msg = str(clustered_df.iloc[0].get("message", ""))
            sample_text = msg[:80] + ("..." if len(msg) > 80 else "")
            stats = {
                "total_clusters": 1,
                "average_cluster_size": 1.0,
                "cluster_distribution": {0: 1},
                "clusters": [
                    {
                        "cluster_id": 0,
                        "theme_label": f"{topic}: {sample_text}",
                        "size": 1,
                        "intra_cluster_similarity": 1.0,
                        "representative_ticket_id": ticket_id,
                        "ticket_ids": [ticket_id],
                    }
                ],
            }
            return clustered_df, stats

        # Generate embeddings if not provided
        if embeddings is None:
            ticket_texts = [
                f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
                for _, row in df.iterrows()
            ]
            embeddings = self.embedding_generator.encode_texts(ticket_texts)

        # Pairwise cosine similarity and distance matrix
        # Cosine distance = 1 - cosine_similarity (bounded [0, 2])
        sim_matrix = np.dot(embeddings, embeddings.T)
        sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
        dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
        np.fill_diagonal(dist_matrix, 0.0)

        # Agglomerative clustering with precomputed distance matrix
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(dist_matrix)

        clustered_df = df.copy()
        clustered_df["cluster_id"] = labels

        # Calculate cluster summaries and statistics
        unique_labels = sorted(list(set(labels)))
        clusters_info = []
        cluster_dist = {}

        for label in unique_labels:
            member_indices = np.where(labels == label)[0]
            cluster_size = len(member_indices)
            cluster_dist[int(label)] = cluster_size

            # Find medoid (ticket with highest average similarity to cluster members)
            if cluster_size == 1:
                medoid_idx = member_indices[0]
                intra_sim = 1.0
            else:
                sub_sim = sim_matrix[np.ix_(member_indices, member_indices)]
                avg_sims = np.mean(sub_sim, axis=1)
                medoid_pos = int(np.argmax(avg_sims))
                medoid_idx = member_indices[medoid_pos]
                # Average non-diagonal similarity
                mask = ~np.eye(cluster_size, dtype=bool)
                intra_sim = float(np.mean(sub_sim[mask]))

            medoid_row = df.iloc[medoid_idx]
            topic = str(medoid_row.get("topic", "Feedback"))
            msg = str(medoid_row.get("message", ""))
            short_msg = msg[:75] + ("..." if len(msg) > 75 else "")
            theme_label = f"{topic}: {short_msg}"

            rep_ticket_id = (
                medoid_row.get("ticket_id", medoid_idx)
                if "ticket_id" in df.columns
                else medoid_idx
            )
            ticket_ids = [
                df.iloc[idx].get("ticket_id", idx) if "ticket_id" in df.columns else idx
                for idx in member_indices
            ]

            clusters_info.append(
                {
                    "cluster_id": int(label),
                    "theme_label": theme_label,
                    "size": cluster_size,
                    "intra_cluster_similarity": round(intra_sim, 3),
                    "representative_ticket_id": rep_ticket_id,
                    "ticket_ids": ticket_ids,
                }
            )

        # Sort clusters by size descending
        clusters_info.sort(key=lambda c: c["size"], reverse=True)

        total_clusters = len(unique_labels)
        avg_size = round(float(np.mean([c["size"] for c in clusters_info])), 2) if clusters_info else 0.0

        stats = {
            "total_clusters": total_clusters,
            "average_cluster_size": avg_size,
            "cluster_distribution": cluster_dist,
            "clusters": clusters_info,
        }

        return clustered_df, stats
