import os
import json
import logging
import numpy as np
import pandas as pd
from src.retrieval import retrieve_tickets
from src.evidence_validation import validate_retrieved_evidence
from src.evidence_deduplication import EvidenceDeduplicator
from src.evidence_clustering import EvidenceClusterer
from src.embedding_generator import EmbeddingGenerator
from src.evaluation.adapters import DatasetRegistry
from src.evaluation.quality import QualityEvaluator

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Empirical Threshold Optimization and Sensitivity Analysis Engine.
    Evaluates evidence validation, deduplication, and clustering parameters across candidate thresholds
    with multi-dataset support and bootstrap confidence intervals.
    """

    @staticmethod
    def compute_bootstrap_ci(
        values: list[float],
        n_bootstraps: int = 1000,
        ci_percentile: float = 95.0,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Compute empirical bootstrap confidence interval for a list of evaluation metric values.
        """
        if not values or len(values) < 2:
            val = float(values[0]) if values else 0.0
            return (round(val, 4), round(val, 4))

        rng = np.random.default_rng(seed)
        boot_means = []
        n = len(values)
        val_arr = np.array(values)

        for _ in range(n_bootstraps):
            sample = rng.choice(val_arr, size=n, replace=True)
            boot_means.append(float(np.mean(sample)))

        alpha = (100.0 - ci_percentile) / 2.0
        lower = float(np.percentile(boot_means, alpha))
        upper = float(np.percentile(boot_means, 100.0 - alpha))
        return (round(lower, 4), round(upper, 4))

    @staticmethod
    def sweep_evidence_similarity_threshold(
        corpus_df: pd.DataFrame | None = None,
        queries: list[dict] | None = None,
        thresholds: list[float] = [0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80],
        ground_truth_path: str = "data/evaluation/retrieval_ground_truth.json",
    ) -> list[dict]:
        """
        Sweep evidence validation similarity thresholds across multi-dataset ground truth.
        """
        if queries is None and os.path.exists(ground_truth_path):
            with open(ground_truth_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)
            queries = gt_data.get("queries", [])

        if not queries:
            return []

        # Group queries by dataset
        queries_by_ds: dict[str, list[dict]] = {}
        for q in queries:
            ds = q.get("dataset", "google_play_reviews")
            if ds not in queries_by_ds:
                queries_by_ds[ds] = []
            queries_by_ds[ds].append(q)

        # Load datasets
        datasets: dict[str, pd.DataFrame] = {}
        for ds in queries_by_ds:
            if corpus_df is not None and ds == "google_play_reviews":
                datasets[ds] = corpus_df
            else:
                df, _ = DatasetRegistry.load_dataset(ds)
                datasets[ds] = df

        results = []
        for thresh in thresholds:
            p_list = []
            r_list = []
            f1_list = []
            retention_list = []

            for ds, q_list in queries_by_ds.items():
                df = datasets[ds]
                for q_item in q_list:
                    q_text = q_item["query"]
                    rel_ids = set(str(t) for t in q_item.get("relevant_ticket_ids", []))
                    if not rel_ids:
                        continue  # Skip negative control queries for precision/recall threshold sweeping

                    matching_tickets, similarity_scores = retrieve_tickets(
                        df, [q_text], top_k=10
                    )
                    val_tickets, _ = validate_retrieved_evidence(
                        matching_tickets, similarity_scores, threshold=thresh, return_scores=True
                    )

                    val_ids = (
                        set(val_tickets["ticket_id"].astype(str).tolist())
                        if not val_tickets.empty
                        else set()
                    )

                    tp = len(val_ids.intersection(rel_ids))
                    fp = len(val_ids - rel_ids)
                    fn = len(rel_ids - val_ids)

                    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1_single = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                    retention = len(val_tickets) / len(matching_tickets) if len(matching_tickets) > 0 else 0.0

                    p_list.append(p)
                    r_list.append(r)
                    f1_list.append(f1_single)
                    retention_list.append(retention)

            mean_p = float(np.mean(p_list)) if p_list else 0.0
            mean_r = float(np.mean(r_list)) if r_list else 0.0
            mean_f1 = float(np.mean(f1_list)) if f1_list else 0.0
            mean_ret = float(np.mean(retention_list)) if retention_list else 0.0

            ci_lower, ci_upper = ThresholdOptimizer.compute_bootstrap_ci(f1_list)

            results.append(
                {
                    "threshold": thresh,
                    "precision": round(mean_p, 4),
                    "recall": round(mean_r, 4),
                    "f1_score": round(mean_f1, 4),
                    "f1_95ci": [ci_lower, ci_upper],
                    "average_retention_rate": round(mean_ret, 4),
                    "queries_evaluated": len(p_list),
                }
            )

        return results

    @staticmethod
    def sweep_deduplication_threshold(
        df: pd.DataFrame | None = None,
        ground_truth_duplicate_pairs: set[tuple[str, str]] | None = None,
        thresholds: list[float] = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
        ground_truth_path: str = "data/evaluation/deduplication_ground_truth.json",
    ) -> list[dict]:
        """
        Sweep deduplication similarity thresholds across multi-dataset ground truth.
        """
        from src.evaluation.deduplication_audit import DeduplicationAuditor
        auditor = DeduplicationAuditor(ground_truth_path=ground_truth_path)

        results = []
        for thresh in thresholds:
            all_tp, all_fp, all_fn, all_tn = 0, 0, 0, 0
            audit_list = auditor.audit_all_datasets(similarity_threshold=thresh)
            for a in audit_list:
                cm = a.get("confusion_matrix", {})
                all_tp += cm.get("true_positives", 0)
                all_fp += cm.get("false_positives", 0)
                all_fn += cm.get("false_negatives", 0)
                all_tn += cm.get("true_negatives", 0)

            p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 1.0
            r = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 1.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            results.append(
                {
                    "threshold": thresh,
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1_score": round(f1, 4),
                    "true_positives": all_tp,
                    "false_positives": all_fp,
                    "false_negatives": all_fn,
                    "true_negatives": all_tn,
                }
            )

        return results

    @staticmethod
    def sweep_clustering_distance_threshold(
        df: pd.DataFrame | None = None,
        distance_thresholds: list[float] = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
        dataset_name: str = "google_play_reviews",
    ) -> list[dict]:
        """
        Sweep clustering distance thresholds and calculate cluster count, coherence, and silhouette.
        """
        if df is None:
            df, _ = DatasetRegistry.load_dataset(dataset_name)

        results = []
        embedder = EmbeddingGenerator()
        ticket_texts = [
            f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
            for _, row in df.iterrows()
        ]
        embeddings = embedder.encode_texts(ticket_texts)

        for dist_thresh in distance_thresholds:
            clusterer = EvidenceClusterer(distance_threshold=dist_thresh)
            clustered_df, stats = clusterer.cluster_tickets(df, embeddings=embeddings)
            eval_res = QualityEvaluator.evaluate_clustering_coherence(
                clustered_df, embeddings
            )

            results.append(
                {
                    "distance_threshold": dist_thresh,
                    "total_clusters": stats["total_clusters"],
                    "average_cluster_size": stats["average_cluster_size"],
                    "silhouette_score": eval_res["silhouette_score"],
                    "intra_cluster_coherence": eval_res["intra_cluster_coherence"],
                }
            )

        return results
