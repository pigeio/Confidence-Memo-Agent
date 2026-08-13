import logging
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from src.retrieval import retrieve_tickets
from src.evidence_deduplication import EvidenceDeduplicator
from src.historical_calibration import HistoricalCalibrationEngine, CalibrationRecord

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """
    Quantitative Quality Evaluation Suite for the Confidence Memo Agent platform.
    Computes precision/recall/MRR for retrieval, precision/recall/F1 for deduplication,
    silhouette coherence for clustering, and Brier/ECE for historical calibration.
    """

    @staticmethod
    def evaluate_retrieval(
        queries: list[dict],
        corpus_df: pd.DataFrame,
        k: int = 5,
    ) -> dict:
        """
        Evaluate semantic retrieval quality over a set of test queries with ground-truth relevance labels.

        Parameters:
            queries (list[dict]): List of query test cases:
                [
                    {
                        "query": "dark mode night theme",
                        "relevant_ticket_ids": {"gp_1001", "gp_1002", "gp_1003", "gp_1004", "gp_1010"}
                    },
                    ...
                ]
            corpus_df (pd.DataFrame): Standardized tickets DataFrame to search.
            k (int): Cutoff rank k for evaluation (default 5).

        Returns:
            dict: {
                "precision_at_k": float,
                "recall_at_k": float,
                "mrr": float,
                "map": float,
                "query_count": int,
                "query_breakdowns": list[dict]
            }
        """
        if not queries:
            return {
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mrr": 0.0,
                "map": 0.0,
                "query_count": 0,
                "query_breakdowns": [],
            }

        p_at_k_list = []
        r_at_k_list = []
        rr_list = []
        ap_list = []
        breakdowns = []

        for q_item in queries:
            query_text = q_item["query"]
            relevant_ids = set(str(tid) for tid in q_item["relevant_ticket_ids"])
            total_relevant = len(relevant_ids)

            # Execute retrieval
            retrieved_df, _ = retrieve_tickets(corpus_df, [query_text], top_k=k)
            retrieved_ids = (
                retrieved_df["ticket_id"].astype(str).tolist()
                if "ticket_id" in retrieved_df.columns
                else []
            )

            # Precision@k
            hits = [1 if tid in relevant_ids else 0 for tid in retrieved_ids]
            p_k = sum(hits) / k if k > 0 else 0.0
            p_at_k_list.append(p_k)

            # Recall@k
            r_k = sum(hits) / total_relevant if total_relevant > 0 else 1.0
            r_at_k_list.append(r_k)

            # Reciprocal Rank (MRR)
            first_hit_rank = 0
            for rank, hit in enumerate(hits, start=1):
                if hit == 1:
                    first_hit_rank = rank
                    break
            rr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
            rr_list.append(rr)

            # Average Precision (AP)
            cum_hits = 0
            precisions = []
            for rank, hit in enumerate(hits, start=1):
                if hit == 1:
                    cum_hits += 1
                    precisions.append(cum_hits / rank)
            ap = sum(precisions) / min(k, total_relevant) if total_relevant > 0 and precisions else 0.0
            ap_list.append(ap)

            breakdowns.append(
                {
                    "query": query_text,
                    "retrieved_count": len(retrieved_ids),
                    "relevant_in_top_k": sum(hits),
                    "total_relevant": total_relevant,
                    "precision_at_k": round(p_k, 3),
                    "recall_at_k": round(r_k, 3),
                    "reciprocal_rank": round(rr, 3),
                    "average_precision": round(ap, 3),
                }
            )

        return {
            "precision_at_k": round(float(np.mean(p_at_k_list)), 4),
            "recall_at_k": round(float(np.mean(r_at_k_list)), 4),
            "mrr": round(float(np.mean(rr_list)), 4),
            "map": round(float(np.mean(ap_list)), 4),
            "query_count": len(queries),
            "query_breakdowns": breakdowns,
        }

    @staticmethod
    def evaluate_deduplication(
        df: pd.DataFrame,
        ground_truth_duplicate_pairs: set[tuple[str, str]],
        similarity_threshold: float = 0.85,
    ) -> dict:
        """
        Evaluate exact + semantic deduplication precision, recall, and F1 score
        against known duplicate ticket pairs.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing duplicate and unique tickets.
            ground_truth_duplicate_pairs (set[tuple[str, str]]): Set of (id_a, id_b) pairs that are true duplicates.
            similarity_threshold (float): Threshold to test.

        Returns:
            dict: {precision, recall, f1, true_positives, false_positives, false_negatives}
        """
        # Normalize ground-truth pairs to ordered tuples (min, max)
        norm_gt = set(
            tuple(sorted([str(p[0]), str(p[1])])) for p in ground_truth_duplicate_pairs
        )

        deduplicator = EvidenceDeduplicator(similarity_threshold=similarity_threshold)
        _, _, stats = deduplicator.deduplicate(df)

        # Reconstruct detected pairs from duplicate groups
        detected_pairs = set()
        for g in stats.get("duplicate_groups", []):
            rep_id = str(g["representative_ticket_id"])
            for dup_id in g["duplicate_ticket_ids"]:
                detected_pairs.add(tuple(sorted([rep_id, str(dup_id)])))

        tp = len(detected_pairs.intersection(norm_gt))
        fp = len(detected_pairs - norm_gt)
        fn = len(norm_gt - detected_pairs)

        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not norm_gt else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "detected_pairs_count": len(detected_pairs),
            "ground_truth_pairs_count": len(norm_gt),
            "deduplication_stats": stats,
        }

    @staticmethod
    def evaluate_clustering_coherence(
        clustered_df: pd.DataFrame,
        embeddings: np.ndarray,
    ) -> dict:
        """
        Evaluate semantic clustering coherence and silhouette score.
        """
        if clustered_df.empty or "cluster_id" not in clustered_df.columns:
            return {"silhouette_score": None, "intra_cluster_coherence": 0.0, "total_clusters": 0}

        labels = clustered_df["cluster_id"].values
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Silhouette score requires 2 <= n_clusters < n_samples
        if 2 <= n_clusters < len(clustered_df):
            try:
                sil_score = float(silhouette_score(embeddings, labels, metric="cosine"))
            except Exception:
                sil_score = None
        else:
            sil_score = None

        # Intra-cluster cosine similarity
        sim_matrix = np.dot(embeddings, embeddings.T)
        intra_sims = []
        for l in unique_labels:
            indices = np.where(labels == l)[0]
            if len(indices) > 1:
                sub_sim = sim_matrix[np.ix_(indices, indices)]
                mask = ~np.eye(len(indices), dtype=bool)
                intra_sims.append(float(np.mean(sub_sim[mask])))
            else:
                intra_sims.append(1.0)

        avg_coherence = float(np.mean(intra_sims)) if intra_sims else 1.0

        return {
            "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
            "intra_cluster_coherence": round(avg_coherence, 4),
            "total_clusters": n_clusters,
        }

    @staticmethod
    def evaluate_calibration(
        records: list[CalibrationRecord],
    ) -> dict:
        """
        Evaluate statistical Brier score and Expected Calibration Error (ECE) for historical calibration.
        """
        engine = HistoricalCalibrationEngine()
        return engine.calculate_calibration_metrics(records=records)
