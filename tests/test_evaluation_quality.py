import pytest
import numpy as np
import pandas as pd
from src.evaluation.quality import QualityEvaluator
from src.historical_calibration import CalibrationRecord


def test_evaluate_retrieval_metrics():
    corpus_df = pd.DataFrame(
        [
            {"ticket_id": "T1", "topic": "Dark Mode", "message": "Add night theme"},
            {"ticket_id": "T2", "topic": "Dark Mode", "message": "Dark theme needed"},
            {"ticket_id": "T3", "topic": "Billing", "message": "Invoice calculation issue"},
            {"ticket_id": "T4", "topic": "Login", "message": "Cannot log in with Google"},
        ]
    )

    queries = [
        {
            "query": "dark mode night theme",
            "relevant_ticket_ids": {"T1", "T2"},
        }
    ]

    metrics = QualityEvaluator.evaluate_retrieval(queries, corpus_df, k=2)
    assert metrics["query_count"] == 1
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert metrics["mrr"] > 0.0
    assert metrics["map"] > 0.0


def test_evaluate_deduplication_metrics():
    df = pd.DataFrame(
        [
            {"ticket_id": "1", "topic": "Theme", "message": "Please add dark mode"},
            {"ticket_id": "2", "topic": "Theme", "message": "Please add dark mode"},  # Exact duplicate
            {"ticket_id": "3", "topic": "Billing", "message": "Charged twice on credit card"},
        ]
    )

    gt_pairs = {("1", "2")}
    metrics = QualityEvaluator.evaluate_deduplication(df, gt_pairs, similarity_threshold=0.85)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_evaluate_clustering_coherence():
    clustered_df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "A", "message": "A1", "cluster_id": 0},
            {"ticket_id": 2, "topic": "A", "message": "A2", "cluster_id": 0},
            {"ticket_id": 3, "topic": "B", "message": "B1", "cluster_id": 1},
            {"ticket_id": 4, "topic": "B", "message": "B2", "cluster_id": 1},
        ]
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.98, 0.05],
            [0.0, 1.0],
            [0.05, 0.98],
        ],
        dtype=np.float32,
    )
    # L2 normalize
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    metrics = QualityEvaluator.evaluate_clustering_coherence(clustered_df, embeddings)
    assert metrics["total_clusters"] == 2
    assert metrics["silhouette_score"] is not None
    assert metrics["silhouette_score"] > 0.5
    assert metrics["intra_cluster_coherence"] > 0.9


def test_evaluate_calibration_metrics():
    records = [
        CalibrationRecord(proposal="P1", predicted_score=90, predicted_confidence="High", predicted_recommendation="PROCEED", actual_outcome=1.0),
        CalibrationRecord(proposal="P2", predicted_score=10, predicted_confidence="Low", predicted_recommendation="REJECT", actual_outcome=0.0),
    ]
    metrics = QualityEvaluator.evaluate_calibration(records)
    assert metrics["evaluated_records"] == 2
    assert metrics["brier_score"] <= 0.05
    assert metrics["status"] == "CALIBRATED"
