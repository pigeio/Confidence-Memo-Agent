import pytest
import numpy as np
import pandas as pd
from src.evaluation.deduplication_audit import DeduplicationAuditor
from src.evaluation.deduplication_candidates import (
    compute_jaccard_similarity,
    compute_char_ngram_similarity,
    DeduplicationCandidateBenchmarker,
)
from src.evaluation.embedding_audit import EmbeddingAuditor
from src.evaluation.model_comparison import ModelComparisonEvaluator
from src.evaluation.threshold_optimizer import ThresholdOptimizer


def test_jaccard_and_ngram_similarity():
    text1 = "Please add dark mode theme option"
    text2 = "Please add dark mode theme setting"
    j_sim = compute_jaccard_similarity(text1, text2)
    assert 0.5 <= j_sim <= 1.0

    ngram_sim = compute_char_ngram_similarity(text1, text2, n=3)
    assert 0.5 <= ngram_sim <= 1.0

    # Completely different
    diff_text = "Unrelated billing invoice error"
    assert compute_jaccard_similarity(text1, diff_text) == 0.0


def test_deduplication_auditor_confusion_matrix():
    auditor = DeduplicationAuditor()
    res = auditor.audit_dataset("google_play_reviews")

    assert res["dataset_name"] == "google_play_reviews"
    cm = res["confusion_matrix"]
    assert cm["true_positives"] >= 0
    assert cm["true_negatives"] >= 0
    assert cm["total_annotated_pairs"] > 0
    assert "precision" in res["metrics"]
    assert "recall" in res["metrics"]
    assert "error_analysis" in res


def test_deduplication_candidate_benchmarker():
    benchmarker = DeduplicationCandidateBenchmarker()
    results = benchmarker.benchmark_candidate_features("google_play_reviews")

    assert len(results) >= 3
    for r in results:
        assert "candidate_name" in r
        assert "precision" in r
        assert "recall" in r
        assert "f1_score" in r


def test_embedding_auditor_batching_and_determinism():
    auditor = EmbeddingAuditor()
    # Batch size sweep
    batch_res = auditor.benchmark_batch_sizes(batch_sizes=[16, 32], sample_size=20)
    assert len(batch_res) == 2
    assert batch_res[0]["throughput_texts_per_sec"] > 0

    # Determinism check
    det_res = auditor.verify_determinism_and_norm(sample_size=10)
    assert det_res["is_deterministic"] is True
    assert det_res["all_l2_unit_normalized"] is True
    assert det_res["norm_violation_count"] == 0

    # Caching check
    cache_res = auditor.verify_caching_performance(sample_size=20)
    assert cache_res["cache_hit_verified"] is True
    assert cache_res["speedup_factor"] > 0


def test_model_comparison_evaluator():
    evaluator = ModelComparisonEvaluator()
    assert len(evaluator.CANDIDATE_MODELS) >= 3
    # Check default model entry
    default_cand = [c for c in evaluator.CANDIDATE_MODELS if c["name"] == "all-MiniLM-L6-v2"][0]
    assert default_cand["dim"] == 384
    assert default_cand["params_millions"] > 20


def test_threshold_optimizer_bootstrap_ci():
    values = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.85]
    ci_lower, ci_upper = ThresholdOptimizer.compute_bootstrap_ci(values, n_bootstraps=500, ci_percentile=95.0)
    assert ci_lower <= np.mean(values) <= ci_upper
    assert 0.70 <= ci_lower <= 0.95
