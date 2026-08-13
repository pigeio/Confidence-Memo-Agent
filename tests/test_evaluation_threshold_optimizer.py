import pytest
import pandas as pd
from src.evaluation.threshold_optimizer import ThresholdOptimizer


@pytest.fixture
def sample_corpus():
    return pd.DataFrame(
        [
            {"ticket_id": "1", "topic": "Theme", "message": "Add dark mode option."},
            {"ticket_id": "2", "topic": "Theme", "message": "The bright white UI hurts my eyes at night."},
            {"ticket_id": "3", "topic": "Billing", "message": "Charged twice on invoice 123."},
        ]
    )


def test_sweep_evidence_similarity_threshold(sample_corpus):
    queries = [
        {
            "query": "dark mode night theme",
            "relevant_ticket_ids": {"1", "2"},
        }
    ]
    results = ThresholdOptimizer.sweep_evidence_similarity_threshold(
        sample_corpus, queries, thresholds=[0.50, 0.60, 0.70]
    )
    assert len(results) == 3
    assert all("threshold" in r for r in results)
    assert all("precision" in r for r in results)
    assert all("recall" in r for r in results)
    assert all("f1_score" in r for r in results)


def test_sweep_deduplication_threshold(sample_corpus):
    gt_pairs = {("1", "2")}
    results = ThresholdOptimizer.sweep_deduplication_threshold(
        sample_corpus, gt_pairs, thresholds=[0.70, 0.85, 0.95]
    )
    assert len(results) == 3
    assert all("threshold" in r for r in results)
    assert all("f1_score" in r for r in results)


def test_sweep_clustering_distance_threshold(sample_corpus):
    results = ThresholdOptimizer.sweep_clustering_distance_threshold(
        sample_corpus, distance_thresholds=[0.25, 0.35, 0.45]
    )
    assert len(results) == 3
    assert all("distance_threshold" in r for r in results)
    assert all("total_clusters" in r for r in results)
