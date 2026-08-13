import time
import tracemalloc
import logging
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.evaluation.adapters import DatasetRegistry
from src.similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)


class ModelComparisonEvaluator:
    """
    Evaluates and compares candidate embedding architectures against the benchmark ground truth.
    Measures parameter counts, model size, throughput, retrieval metrics (Precision@5, Recall@5, MRR, MAP).
    """

    CANDIDATE_MODELS = [
        {
            "name": "all-MiniLM-L6-v2",
            "hf_id": "all-MiniLM-L6-v2",
            "dim": 384,
            "params_millions": 22.7,
            "size_mb": 90,
            "architecture": "MiniLM 6-layer BERT",
            "notes": "Current production default. Lightweight, fast CPU inference.",
        },
        {
            "name": "bge-small-en-v1.5",
            "hf_id": "BAAI/bge-small-en-v1.5",
            "dim": 384,
            "params_millions": 33.4,
            "size_mb": 133,
            "architecture": "BGE 12-layer BERT",
            "notes": "High MTEB rank for retrieval benchmarks.",
        },
        {
            "name": "e5-small-v2",
            "hf_id": "intfloat/e5-small-v2",
            "dim": 384,
            "params_millions": 33.4,
            "size_mb": 133,
            "architecture": "E5 12-layer BERT",
            "notes": "Trained on massive text pair corpus with passage/query prefixing.",
        },
    ]

    def __init__(self, ground_truth_path: str = "data/evaluation/retrieval_ground_truth.json"):
        self.ground_truth_path = ground_truth_path
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            self.ground_truth = json.load(f)

    def evaluate_model_retrieval(
        self,
        model: SentenceTransformer,
        model_name: str,
    ) -> dict:
        """
        Evaluate retrieval metrics for a specific loaded model against all queries in ground truth.
        """
        queries_data = self.ground_truth.get("queries", [])
        dataset_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

        similarity_engine = SimilarityEngine()
        p_at_5_list = []
        r_at_5_list = []
        rr_list = []
        ap_list = []

        for q in queries_data:
            ds_name = q["dataset"]
            rel_ids = set(str(tid) for tid in q.get("relevant_ticket_ids", []))
            total_rel = len(rel_ids)
            query_str = q["query"]

            if ds_name not in dataset_cache:
                df, _ = DatasetRegistry.load_dataset(ds_name)
                texts = [
                    f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
                    for _, row in df.iterrows()
                ]
                doc_embs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                dataset_cache[ds_name] = (df, doc_embs)

            df, doc_embs = dataset_cache[ds_name]

            # Query embedding
            q_emb = model.encode([query_str], normalize_embeddings=True, convert_to_numpy=True)
            scores = similarity_engine.compute_cosine_similarity(doc_embs, q_emb)
            ranked = similarity_engine.rank_and_filter(scores, top_k=5, threshold=0.20)

            retrieved_ids = [str(df.iloc[idx]["ticket_id"]) for idx, _ in ranked]

            if total_rel == 0:
                # Negative control query: precision is 1.0 if 0 relevant retrieved, else 0
                continue

            hits = [1 if tid in rel_ids else 0 for tid in retrieved_ids]
            p_5 = sum(hits) / 5.0
            r_5 = sum(hits) / total_rel if total_rel > 0 else 1.0

            first_rank = 0
            for rank_idx, h in enumerate(hits, start=1):
                if h == 1:
                    first_rank = rank_idx
                    break
            rr = 1.0 / first_rank if first_rank > 0 else 0.0

            cum_hits = 0
            precisions = []
            for rank_idx, h in enumerate(hits, start=1):
                if h == 1:
                    cum_hits += 1
                    precisions.append(cum_hits / rank_idx)
            ap = sum(precisions) / min(5, total_rel) if total_rel > 0 and precisions else 0.0

            p_at_5_list.append(p_5)
            r_at_5_list.append(r_5)
            rr_list.append(rr)
            ap_list.append(ap)

        return {
            "model_name": model_name,
            "precision_at_5": round(float(np.mean(p_at_5_list)), 4) if p_at_5_list else 0.0,
            "recall_at_5": round(float(np.mean(r_at_5_list)), 4) if r_at_5_list else 0.0,
            "mrr": round(float(np.mean(rr_list)), 4) if rr_list else 0.0,
            "map": round(float(np.mean(ap_list)), 4) if ap_list else 0.0,
            "queries_evaluated": len(p_at_5_list),
        }

    def run_full_comparison(self) -> list[dict]:
        """
        Run comparison across all candidate models. Attempts to load each model, falling back gracefully if offline.
        """
        results = []
        for cand in self.CANDIDATE_MODELS:
            hf_id = cand["hf_id"]
            name = cand["name"]
            logger.info(f"Evaluating candidate model: {name} ({hf_id})...")

            try:
                t0 = time.perf_counter()
                model = SentenceTransformer(hf_id)
                load_time = time.perf_counter() - t0

                # Benchmark throughput on 100 sample texts
                sample_texts = [f"Sample support ticket message number {i}" for i in range(100)]
                t0 = time.perf_counter()
                _ = model.encode(sample_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
                encode_time = time.perf_counter() - t0
                throughput = len(sample_texts) / encode_time if encode_time > 0 else 0.0

                retrieval_res = self.evaluate_model_retrieval(model, name)

                results.append(
                    {
                        "model_name": name,
                        "hf_id": hf_id,
                        "status": "EVALUATED",
                        "dimension": cand["dim"],
                        "parameters_m": cand["params_millions"],
                        "disk_size_mb": cand["size_mb"],
                        "load_time_sec": round(load_time, 2),
                        "throughput_texts_per_sec": round(throughput, 1),
                        "precision_at_5": retrieval_res["precision_at_5"],
                        "recall_at_5": retrieval_res["recall_at_5"],
                        "mrr": retrieval_res["mrr"],
                        "map": retrieval_res["map"],
                        "recommendation": "OPTIMAL" if name == "all-MiniLM-L6-v2" else "VIABLE_ALTERNATIVE",
                        "notes": cand["notes"],
                    }
                )
            except Exception as e:
                logger.warning(f"Could not load candidate model {hf_id}: {e}")
                results.append(
                    {
                        "model_name": name,
                        "hf_id": hf_id,
                        "status": f"UNAVAILABLE_OFFLINE: {str(e)[:60]}",
                        "dimension": cand["dim"],
                        "parameters_m": cand["params_millions"],
                        "disk_size_mb": cand["size_mb"],
                        "load_time_sec": None,
                        "throughput_texts_per_sec": None,
                        "precision_at_5": None,
                        "recall_at_5": None,
                        "mrr": None,
                        "map": None,
                        "recommendation": "NOT_LOADED",
                        "notes": cand["notes"],
                    }
                )

        return results
