import os
import json
import logging
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from src.evidence_deduplication import EvidenceDeduplicator
from src.evaluation.adapters import DatasetRegistry

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationErrorItem:
    ticket_a: str
    ticket_b: str
    message_a: str
    message_b: str
    predicted_duplicate: bool
    ground_truth_duplicate: bool
    error_type: str  # "FALSE_POSITIVE" or "FALSE_NEGATIVE"
    similarity_score: float
    classified_cause: str
    justification: str

    def to_dict(self) -> dict:
        return asdict(self)


class DeduplicationAuditor:
    """
    Independent Deduplication Audit Engine.
    Computes complete confusion matrices (TP, FP, FN, TN), detailed error analysis,
    failure cause classification, and score distribution diagnostics.
    """

    CAUSE_CATEGORIES = [
        "generic_wording",
        "different_intent",
        "different_feature",
        "short_message",
        "embedding_confusion",
        "threshold_too_low",
        "threshold_too_high",
        "incomplete_ground_truth",
        "unknown",
    ]

    def __init__(self, ground_truth_path: str = "data/evaluation/deduplication_ground_truth.json"):
        self.ground_truth_path = ground_truth_path
        self._ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> dict:
        if not os.path.exists(self.ground_truth_path):
            raise FileNotFoundError(f"Deduplication ground truth not found at {self.ground_truth_path}")
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _classify_error_cause(
        msg_a: str,
        msg_b: str,
        sim: float,
        error_type: str,
        threshold: float,
    ) -> str:
        """Classify root cause for misclassified duplicate pairs."""
        len_a = len(msg_a.strip())
        len_b = len(msg_b.strip())

        if error_type == "FALSE_POSITIVE":
            if sim >= threshold and sim < threshold + 0.05:
                return "threshold_too_low"
            if len_a < 30 or len_b < 30:
                return "short_message"
            # Check for generic keywords (app, broken, issue, fix)
            common_words = set(msg_a.lower().split()).intersection(set(msg_b.lower().split()))
            generic_stop = {"the", "a", "is", "app", "to", "for", "please", "in", "and", "my", "on", "it"}
            meaningful_common = common_words - generic_stop
            if len(meaningful_common) <= 1:
                return "embedding_confusion"
            return "different_intent"

        if error_type == "FALSE_NEGATIVE":
            if sim < threshold and sim >= threshold - 0.10:
                return "threshold_too_high"
            if len_a < 30 or len_b < 30:
                return "short_message"
            return "embedding_confusion"

        return "unknown"

    def audit_dataset(
        self,
        dataset_name: str,
        similarity_threshold: float = 0.85,
    ) -> dict:
        """
        Perform complete audit of duplicate detection on a given dataset.
        """
        df, meta = DatasetRegistry.load_dataset(dataset_name)
        gt_data = self._ground_truth.get("datasets", {}).get(dataset_name, {})
        gt_pairs_list = gt_data.get("pairs", [])

        if not gt_pairs_list:
            return {
                "dataset_name": dataset_name,
                "status": "NO_GROUND_TRUTH",
                "total_annotated_pairs": 0,
            }

        # Build ground-truth lookup: (min_id, max_id) -> is_duplicate bool
        gt_map: dict[tuple[str, str], dict] = {}
        for p in gt_pairs_list:
            key = tuple(sorted([str(p["ticket_a"]), str(p["ticket_b"])]))
            gt_map[key] = p

        # Run deduplication
        deduplicator = EvidenceDeduplicator(similarity_threshold=similarity_threshold)
        dedup_df, _, stats = deduplicator.deduplicate(df)

        # Reconstruct detected duplicate pairs from disjoint groups
        detected_pairs: set[tuple[str, str]] = set()
        for g in stats.get("duplicate_groups", []):
            rep_id = str(g["representative_ticket_id"])
            for dup_id in g["duplicate_ticket_ids"]:
                detected_pairs.add(tuple(sorted([rep_id, str(dup_id)])))

        # Also obtain embedding similarity matrix for exact pair similarity scores
        ticket_texts = [
            f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
            for _, row in df.iterrows()
        ]
        embeddings = deduplicator.embedding_generator.encode_texts(ticket_texts)
        sim_matrix = np.dot(embeddings, embeddings.T)

        id_to_idx = {str(row.get("ticket_id", i)): i for i, row in df.iterrows()}

        # Evaluate all annotated pairs
        tp, fp, fn, tn = 0, 0, 0, 0
        error_items: list[DeduplicationErrorItem] = []
        cause_counts: dict[str, int] = {c: 0 for c in self.CAUSE_CATEGORIES}

        tp_pairs = []
        tn_pairs = []

        for pair_key, p_info in gt_map.items():
            id_a, id_b = pair_key
            is_gt_dup = p_info["is_duplicate"]
            is_pred_dup = pair_key in detected_pairs

            idx_a = id_to_idx.get(id_a)
            idx_b = id_to_idx.get(id_b)
            sim_score = float(sim_matrix[idx_a, idx_b]) if (idx_a is not None and idx_b is not None) else 0.0

            msg_a = str(df.iloc[idx_a]["message"]) if idx_a is not None else ""
            msg_b = str(df.iloc[idx_b]["message"]) if idx_b is not None else ""

            if is_gt_dup and is_pred_dup:
                tp += 1
                tp_pairs.append({"pair": list(pair_key), "similarity": round(sim_score, 4)})
            elif not is_gt_dup and not is_pred_dup:
                tn += 1
                tn_pairs.append({"pair": list(pair_key), "similarity": round(sim_score, 4)})
            elif not is_gt_dup and is_pred_dup:
                fp += 1
                cause = self._classify_error_cause(msg_a, msg_b, sim_score, "FALSE_POSITIVE", similarity_threshold)
                cause_counts[cause] += 1
                error_items.append(
                    DeduplicationErrorItem(
                        ticket_a=id_a,
                        ticket_b=id_b,
                        message_a=msg_a,
                        message_b=msg_b,
                        predicted_duplicate=True,
                        ground_truth_duplicate=False,
                        error_type="FALSE_POSITIVE",
                        similarity_score=round(sim_score, 4),
                        classified_cause=cause,
                        justification=p_info.get("justification", ""),
                    )
                )
            elif is_gt_dup and not is_pred_dup:
                fn += 1
                cause = self._classify_error_cause(msg_a, msg_b, sim_score, "FALSE_NEGATIVE", similarity_threshold)
                cause_counts[cause] += 1
                error_items.append(
                    DeduplicationErrorItem(
                        ticket_a=id_a,
                        ticket_b=id_b,
                        message_a=msg_a,
                        message_b=msg_b,
                        predicted_duplicate=False,
                        ground_truth_duplicate=True,
                        error_type="FALSE_NEGATIVE",
                        similarity_score=round(sim_score, 4),
                        classified_cause=cause,
                        justification=p_info.get("justification", ""),
                    )
                )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 1.0

        return {
            "dataset_name": dataset_name,
            "similarity_threshold": similarity_threshold,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "total_annotated_pairs": len(gt_map),
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "specificity": round(specificity, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(accuracy, 4),
            },
            "error_analysis": {
                "total_errors": len(error_items),
                "false_positives_count": fp,
                "false_negatives_count": fn,
                "cause_distribution": {k: v for k, v in cause_counts.items() if v > 0},
                "error_details": [e.to_dict() for e in error_items],
            },
            "correct_predictions": {
                "true_positives_count": tp,
                "true_negatives_count": tn,
            },
        }

    def audit_all_datasets(self, similarity_threshold: float = 0.85) -> list[dict]:
        results = []
        for d in self._ground_truth.get("datasets", {}):
            results.append(self.audit_dataset(d, similarity_threshold=similarity_threshold))
        return results
