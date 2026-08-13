import re
import string
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.evaluation.adapters import DatasetRegistry
from src.evaluation.deduplication_audit import DeduplicationAuditor

logger = logging.getLogger(__name__)


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute token-level Jaccard similarity between two texts after normalization."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union) if union else 0.0


def compute_char_ngram_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    clean_a = re.sub(r"\s+", " ", text_a.lower().strip())
    clean_b = re.sub(r"\s+", " ", text_b.lower().strip())
    if len(clean_a) < n or len(clean_b) < n:
        return 1.0 if clean_a == clean_b else 0.0
    ngrams_a = set(clean_a[i : i + n] for i in range(len(clean_a) - n + 1))
    ngrams_b = set(clean_b[i : i + n] for i in range(len(clean_b) - n + 1))
    intersection = ngrams_a.intersection(ngrams_b)
    union = ngrams_a.union(ngrams_b)
    return len(intersection) / len(union) if union else 0.0


class DeduplicationCandidateBenchmarker:
    """
    Evaluates candidate deterministic filters for improving deduplication precision.
    Tests candidate features (Jaccard, N-grams, Topic metadata) independently without altering production logic.
    """

    def __init__(self, ground_truth_path: str = "data/evaluation/deduplication_ground_truth.json"):
        self.auditor = DeduplicationAuditor(ground_truth_path=ground_truth_path)

    def benchmark_candidate_features(
        self,
        dataset_name: str = "google_play_reviews",
    ) -> list[dict]:
        """
        Benchmark different deterministic filter combinations against annotated ground truth.
        """
        df, _ = DatasetRegistry.load_dataset(dataset_name)
        gt_data = self.auditor._ground_truth.get("datasets", {}).get(dataset_name, {})
        gt_pairs = gt_data.get("pairs", [])

        if not gt_pairs:
            return []

        id_to_row = {str(row["ticket_id"]): row for _, row in df.iterrows()}
        candidate_configs = [
            {"name": "Baseline (Embedding Cosine >= 0.85)", "embed_thresh": 0.85, "jaccard_thresh": 0.0, "ngram_thresh": 0.0, "require_topic_match": False},
            {"name": "Cosine >= 0.85 + Jaccard >= 0.20", "embed_thresh": 0.85, "jaccard_thresh": 0.20, "ngram_thresh": 0.0, "require_topic_match": False},
            {"name": "Cosine >= 0.85 + Jaccard >= 0.30", "embed_thresh": 0.85, "jaccard_thresh": 0.30, "ngram_thresh": 0.0, "require_topic_match": False},
            {"name": "Cosine >= 0.85 + 3-gram >= 0.25", "embed_thresh": 0.85, "jaccard_thresh": 0.0, "ngram_thresh": 0.25, "require_topic_match": False},
            {"name": "Cosine >= 0.80 + Jaccard >= 0.30", "embed_thresh": 0.80, "jaccard_thresh": 0.30, "ngram_thresh": 0.0, "require_topic_match": False},
            {"name": "Cosine >= 0.85 + Exact Topic Match", "embed_thresh": 0.85, "jaccard_thresh": 0.0, "ngram_thresh": 0.0, "require_topic_match": True},
        ]

        from src.embedding_generator import EmbeddingGenerator
        embedder = EmbeddingGenerator()
        texts = [f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}" for _, row in df.iterrows()]
        embeddings = embedder.encode_texts(texts)
        sim_matrix = np.dot(embeddings, embeddings.T)
        id_to_idx = {str(row["ticket_id"]): i for i, row in df.iterrows()}

        results = []
        for cfg in candidate_configs:
            tp, fp, fn, tn = 0, 0, 0, 0
            for p in gt_pairs:
                id_a, id_b = str(p["ticket_a"]), str(p["ticket_b"])
                is_gt = p["is_duplicate"]

                row_a = id_to_row.get(id_a)
                row_b = id_to_row.get(id_b)
                if row_a is None or row_b is None:
                    continue

                idx_a = id_to_idx[id_a]
                idx_b = id_to_idx[id_b]
                cos_sim = float(sim_matrix[idx_a, idx_b])

                msg_a = str(row_a.get("message", ""))
                msg_b = str(row_b.get("message", ""))

                jaccard = compute_jaccard_similarity(msg_a, msg_b)
                ngram = compute_char_ngram_similarity(msg_a, msg_b, n=3)
                topic_match = str(row_a.get("topic", "")).strip().lower() == str(row_b.get("topic", "")).strip().lower()

                # Decision rule
                pred = cos_sim >= cfg["embed_thresh"]
                if cfg["jaccard_thresh"] > 0:
                    pred = pred and (jaccard >= cfg["jaccard_thresh"])
                if cfg["ngram_thresh"] > 0:
                    pred = pred and (ngram >= cfg["ngram_thresh"])
                if cfg["require_topic_match"]:
                    pred = pred and topic_match

                if is_gt and pred:
                    tp += 1
                elif not is_gt and not pred:
                    tn += 1
                elif not is_gt and pred:
                    fp += 1
                elif is_gt and not pred:
                    fn += 1

            p_val = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1_val = (2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0

            results.append(
                {
                    "candidate_name": cfg["name"],
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "true_negatives": tn,
                    "precision": round(p_val, 4),
                    "recall": round(r_val, 4),
                    "f1_score": round(f1_val, 4),
                }
            )

        return results
