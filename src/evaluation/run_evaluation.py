import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
from src.evaluation.adapters import DatasetRegistry
from src.evaluation.deduplication_audit import DeduplicationAuditor
from src.evaluation.deduplication_candidates import DeduplicationCandidateBenchmarker
from src.evaluation.embedding_audit import EmbeddingAuditor
from src.evaluation.model_comparison import ModelComparisonEvaluator
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.similarity_engine import SimilarityEngine
from src.embedding_generator import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_comprehensive_retrieval_audit(
    ground_truth_path: str = "data/evaluation/retrieval_ground_truth.json",
) -> dict:
    """
    Evaluate retrieval quality across all 20+ multi-domain queries in the ground truth dataset.
    """
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    queries = gt_data.get("queries", [])
    embedder_model = get_embedding_model()
    sim_engine = SimilarityEngine()

    dataset_cache: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    p_at_5_list = []
    r_at_5_list = []
    rr_list = []
    ap_list = []
    neg_control_results = []
    breakdowns = []

    for q_item in queries:
        ds_name = q_item["dataset"]
        rel_ids = set(str(t) for t in q_item.get("relevant_ticket_ids", []))
        q_text = q_item["query"]
        q_type = q_item.get("query_type", "standard")
        q_id = q_item.get("query_id", "")

        if ds_name not in dataset_cache:
            df, _ = DatasetRegistry.load_dataset(ds_name)
            texts = [f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}" for _, row in df.iterrows()]
            embs = embedder_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            dataset_cache[ds_name] = (df, embs)

        df, doc_embs = dataset_cache[ds_name]

        # Query vector search
        q_emb = embedder_model.encode([q_text], normalize_embeddings=True, convert_to_numpy=True)
        scores = sim_engine.compute_cosine_similarity(doc_embs, q_emb)
        ranked = sim_engine.rank_and_filter(scores, top_k=5, threshold=0.20)

        retrieved_ids = [str(df.iloc[idx]["ticket_id"]) for idx, _ in ranked]
        retrieved_scores = [float(s) for _, s in ranked]

        if q_type == "negative_control":
            # Check how many pass evidence threshold (0.60)
            false_positives_above_60 = sum(1 for s in retrieved_scores if s >= 0.60)
            neg_control_results.append(
                {
                    "query_id": q_id,
                    "query": q_text,
                    "dataset": ds_name,
                    "top_similarity_score": round(max(retrieved_scores), 4) if retrieved_scores else 0.0,
                    "false_positives_above_evidence_threshold": false_positives_above_60,
                    "properly_rejected": bool(false_positives_above_60 == 0),
                }
            )
            continue

        hits = [1 if tid in rel_ids else 0 for tid in retrieved_ids]
        total_rel = len(rel_ids)

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

        breakdowns.append(
            {
                "query_id": q_id,
                "dataset": ds_name,
                "query": q_text,
                "query_type": q_type,
                "precision_at_5": round(p_5, 3),
                "recall_at_5": round(r_5, 3),
                "mrr": round(rr, 3),
                "map": round(ap, 3),
                "retrieved_count": len(retrieved_ids),
                "relevant_retrieved": sum(hits),
                "total_relevant": total_rel,
            }
        )

    return {
        "total_queries": len(queries),
        "evaluated_positive_queries": len(p_at_5_list),
        "negative_control_queries": len(neg_control_results),
        "macro_precision_at_5": round(float(np.mean(p_at_5_list)), 4) if p_at_5_list else 0.0,
        "macro_recall_at_5": round(float(np.mean(r_at_5_list)), 4) if r_at_5_list else 0.0,
        "mean_reciprocal_rank": round(float(np.mean(rr_list)), 4) if rr_list else 0.0,
        "mean_average_precision": round(float(np.mean(ap_list)), 4) if ap_list else 0.0,
        "negative_control_audit": neg_control_results,
        "per_query_breakdowns": breakdowns,
    }


def generate_markdown_audit_report(data: dict, out_path: str = "docs/audit/evaluation_results.md") -> None:
    """
    Generate comprehensive markdown audit document from computed metrics.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ret = data["retrieval_audit"]
    dedup = data["deduplication_audit"]
    emb = data["embedding_audit"]
    models = data["model_comparison"]
    thresh = data["threshold_optimization"]
    cand = data["candidate_benchmarks"]

    lines = [
        "# Scientific Evaluation Audit & Benchmark Results",
        "",
        "This report documents the verified empirical measurements, confusion matrices, failure classifications, and model comparisons produced by the independent evaluation audit suite.",
        "",
        "---",
        "",
        "## 1. Retrieval Evaluation Audit",
        "",
        f"- **Total Test Queries:** {ret['total_queries']} ({ret['evaluated_positive_queries']} Positive, {ret['negative_control_queries']} Negative Controls)",
        f"- **Precision@5 (Macro):** `{ret['macro_precision_at_5']:.4f}`",
        f"- **Recall@5 (Macro):** `{ret['macro_recall_at_5']:.4f}`",
        f"- **Mean Reciprocal Rank (MRR):** `{ret['mean_reciprocal_rank']:.4f}`",
        f"- **Mean Average Precision (MAP):** `{ret['mean_average_precision']:.4f}`",
        "",
        "### Negative Control Resistance (False Positive Shielding):",
        "| Query ID | Dataset | Query | Top Similarity Score | Above 0.60 Threshold | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for n in ret.get("negative_control_audit", []):
        status = "PASSED (Rejected)" if n["properly_rejected"] else "FAILED (False Positive)"
        lines.append(
            f"| `{n['query_id']}` | `{n['dataset']}` | *{n['query']}* | `{n['top_similarity_score']:.4f}` | `{n['false_positives_above_evidence_threshold']}` | **{status}** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Deduplication Audit & Confusion Matrix",
            "",
            "| Dataset | TP | FP | FN | TN | Precision | Recall | Specificity | F1 Score | Accuracy |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for d in dedup:
        cm = d["confusion_matrix"]
        m = d["metrics"]
        lines.append(
            f"| `{d['dataset_name']}` | {cm['true_positives']} | {cm['false_positives']} | {cm['false_negatives']} | {cm['true_negatives']} | `{m['precision']:.3f}` | `{m['recall']:.3f}` | `{m['specificity']:.3f}` | `{m['f1_score']:.3f}` | `{m['accuracy']:.3f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Deduplication Candidate Filters Benchmark",
            "",
            "Evaluating whether secondary deterministic filters (Jaccard, N-grams, Topic) improve precision without harming recall:",
            "",
            "| Candidate Feature Pipeline | TP | FP | FN | TN | Precision | Recall | F1 Score |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for c in cand:
        lines.append(
            f"| {c['candidate_name']} | {c['true_positives']} | {c['false_positives']} | {c['false_negatives']} | {c['true_negatives']} | `{c['precision']:.3f}` | `{c['recall']:.3f}` | `{c['f1_score']:.3f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Embedding Generation Performance Audit",
            "",
            "### Batch Size Scaling on CPU:",
            "| Batch Size | Sample Size | Elapsed Time | Throughput | Peak RAM | Latency / Text |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for b in emb.get("batch_sizes", []):
        lines.append(
            f"| `{b['batch_size']}` | {b['sample_size']} | `{b['elapsed_seconds']:.3f}s` | **{b['throughput_texts_per_sec']} texts/s** | `{b['peak_memory_mb']} MB` | `{b['avg_ms_per_text']} ms` |"
        )

    det = emb.get("determinism", {})
    cache = emb.get("caching", {})
    lines.extend(
        [
            "",
            "### Determinism & Caching Verification:",
            f"- **Strict Numerical Determinism:** `{det.get('is_deterministic')}` (Max $\\Delta = {det.get('max_absolute_difference', 0.0):.1e}$)",
            f"- **All Vectors $L_2$ Normalized:** `{det.get('all_l2_unit_normalized')}` (Norm Violations: `{det.get('norm_violation_count')}`)",
            f"- **Cache Speedup Factor:** `{cache.get('speedup_factor', 0.0)}x` (`{cache.get('cold_start_seconds')}s` cold vs `{cache.get('cached_lookup_seconds')}s` cached)",
            "",
            "---",
            "",
            "## 5. Embedding Model Architecture Comparison",
            "",
            "| Model Architecture | Parameters | Disk Size | Throughput (CPU) | Precision@5 | Recall@5 | MRR | Recommendation |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for m in models:
        p5 = f"`{m['precision_at_5']:.3f}`" if m["precision_at_5"] is not None else "N/A"
        r5 = f"`{m['recall_at_5']:.3f}`" if m["recall_at_5"] is not None else "N/A"
        mrr = f"`{m['mrr']:.3f}`" if m["mrr"] is not None else "N/A"
        tp = f"{m['throughput_texts_per_sec']} txt/s" if m["throughput_texts_per_sec"] is not None else "N/A"
        lines.append(
            f"| `{m['model_name']}` | {m['parameters_m']}M | {m['disk_size_mb']} MB | {tp} | {p5} | {r5} | {mrr} | **{m['recommendation']}** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Threshold Revalidation & 95% Confidence Intervals",
            "",
            "### Evidence Similarity Threshold Sweep (Multi-Dataset, Bootstrap N=1000):",
            "| Threshold | Precision | Recall | F1 Score | F1 95% Confidence Interval | Retention Rate |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for s in thresh.get("evidence_threshold_sweep", []):
        ci = s.get("f1_95ci", [0.0, 0.0])
        lines.append(
            f"| `{s['threshold']:.2f}` | `{s['precision']:.3f}` | `{s['recall']:.3f}` | `{s['f1_score']:.3f}` | `[{ci[0]:.3f}, {ci[1]:.3f}]` | `{s['average_retention_rate']*100:.1f}%` |"
        )

    lines.extend(
        [
            "",
            "### Deduplication Threshold Sweep (Confusion Matrix):",
            "| Threshold | TP | FP | FN | TN | Precision | Recall | F1 Score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for d in thresh.get("deduplication_threshold_sweep", []):
        lines.append(
            f"| `{d['threshold']:.2f}` | {d['true_positives']} | {d['false_positives']} | {d['false_negatives']} | {d['true_negatives']} | `{d['precision']:.3f}` | `{d['recall']:.3f}` | `{d['f1_score']:.3f}` |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Generated markdown audit report at: {out_path}")


def main():
    set_seed(42)
    logger.info("=== STARTING INDEPENDENT SCIENTIFIC AUDIT EVALUATION RUNNER ===")

    # 1. Retrieval Audit
    logger.info("[1/6] Running Retrieval Quality Audit on Ground Truth...")
    retrieval_res = run_comprehensive_retrieval_audit()

    # 2. Deduplication Audit
    logger.info("[2/6] Running Deduplication Confusion Matrix & Error Audit...")
    dedup_auditor = DeduplicationAuditor()
    dedup_res = dedup_auditor.audit_all_datasets(similarity_threshold=0.85)

    # 3. Candidate Filters Benchmark
    logger.info("[3/6] Benchmarking Deduplication Candidate Features...")
    cand_benchmarker = DeduplicationCandidateBenchmarker()
    cand_res = cand_benchmarker.benchmark_candidate_features("google_play_reviews")

    # 4. Embedding Performance Audit
    logger.info("[4/6] Auditing Embedding Generation & Batch Sizes...")
    emb_auditor = EmbeddingAuditor()
    batch_res = emb_auditor.benchmark_batch_sizes(batch_sizes=[16, 32, 64, 128, 256], sample_size=500)
    det_res = emb_auditor.verify_determinism_and_norm(sample_size=50)
    cache_res = emb_auditor.verify_caching_performance(sample_size=200)
    emb_res = {
        "batch_sizes": batch_res,
        "determinism": det_res,
        "caching": cache_res,
    }

    # 5. Model Architecture Comparison
    logger.info("[5/6] Comparing Candidate Embedding Models...")
    comp_evaluator = ModelComparisonEvaluator()
    model_res = comp_evaluator.run_full_comparison()

    # 6. Threshold Revalidation
    logger.info("[6/6] Revalidating Thresholds with Multi-Dataset Ground Truth & Bootstrap CI...")
    ev_sweep = ThresholdOptimizer.sweep_evidence_similarity_threshold(
        thresholds=[0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    )
    dedup_sweep = ThresholdOptimizer.sweep_deduplication_threshold(
        thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    cluster_sweep = ThresholdOptimizer.sweep_clustering_distance_threshold(
        distance_thresholds=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    )
    thresh_res = {
        "evidence_threshold_sweep": ev_sweep,
        "deduplication_threshold_sweep": dedup_sweep,
        "clustering_threshold_sweep": cluster_sweep,
    }

    # Compile JSON bundle
    full_data = {
        "audit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "retrieval_audit": retrieval_res,
        "deduplication_audit": dedup_res,
        "candidate_benchmarks": cand_res,
        "embedding_audit": emb_res,
        "model_comparison": model_res,
        "threshold_optimization": thresh_res,
    }

    out_json = "data/evaluation/audit_metrics.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_data, f, indent=2)
    logger.info(f"Saved audit metrics to {out_json}")

    # Generate markdown report
    generate_markdown_audit_report(full_data, "docs/audit/evaluation_results.md")
    logger.info("=== AUDIT COMPLETE ===")


if __name__ == "__main__":
    main()
