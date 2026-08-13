import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from src.evidence_clustering import EvidenceClusterer


def test_clusterer_initialization():
    clusterer = EvidenceClusterer(distance_threshold=0.4)
    assert clusterer.distance_threshold == 0.4

    with pytest.raises(ValueError, match="distance_threshold must be between"):
        EvidenceClusterer(distance_threshold=-0.1)

    with pytest.raises(ValueError, match="distance_threshold must be between"):
        EvidenceClusterer(distance_threshold=2.5)


def test_clustering_empty_and_single_df():
    clusterer = EvidenceClusterer()

    # Empty df
    empty_df = pd.DataFrame(columns=["topic", "message"])
    res_df, stats = clusterer.cluster_tickets(empty_df)
    assert res_df.empty
    assert "cluster_id" in res_df.columns
    assert stats["total_clusters"] == 0
    assert stats["average_cluster_size"] == 0.0

    # Single df
    single_df = pd.DataFrame(
        [{"ticket_id": "T1", "topic": "Login", "message": "Can't log in"}]
    )
    res_df, stats = clusterer.cluster_tickets(single_df)
    assert len(res_df) == 1
    assert res_df.iloc[0]["cluster_id"] == 0
    assert stats["total_clusters"] == 1
    assert stats["average_cluster_size"] == 1.0
    assert stats["clusters"][0]["representative_ticket_id"] == "T1"


def test_clustering_errors():
    clusterer = EvidenceClusterer()

    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        clusterer.cluster_tickets("invalid_df")

    with pytest.raises(TypeError, match="embeddings must be a numpy ndarray"):
        clusterer.cluster_tickets(
            pd.DataFrame([{"topic": "A", "message": "B"}]),
            embeddings="not_an_array",
        )

    with pytest.raises(ValueError, match="embeddings length .* does not match"):
        clusterer.cluster_tickets(
            pd.DataFrame([{"topic": "A", "message": "B"}]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]]),
        )

    with pytest.raises(ValueError, match="df is missing required columns"):
        clusterer.cluster_tickets(
            pd.DataFrame([{"foo": "bar"}, {"foo": "baz"}])
        )


def test_semantic_clustering_distinct_groups():
    # 4 tickets: 2 about Dark Theme, 2 about Billing
    df = pd.DataFrame(
        [
            {"ticket_id": 1, "topic": "Theme", "message": "Please support dark mode theme"},
            {"ticket_id": 2, "topic": "Theme", "message": "Night theme is essential for dark rooms"},
            {"ticket_id": 3, "topic": "Billing", "message": "Invoice calculation error on renewal"},
            {"ticket_id": 4, "topic": "Billing", "message": "Charged incorrectly for monthly subscription"},
        ]
    )

    # 2 distinct cluster embeddings
    # Theme vectors close to [1, 0]
    v1 = np.array([0.98, 0.05], dtype=np.float32)
    v1 /= np.linalg.norm(v1)
    v2 = np.array([0.95, 0.08], dtype=np.float32)
    v2 /= np.linalg.norm(v2)
    # Billing vectors close to [0, 1]
    v3 = np.array([0.05, 0.98], dtype=np.float32)
    v3 /= np.linalg.norm(v3)
    v4 = np.array([0.08, 0.95], dtype=np.float32)
    v4 /= np.linalg.norm(v4)

    mock_embeddings = np.vstack([v1, v2, v3, v4])

    clusterer = EvidenceClusterer(distance_threshold=0.35)
    res_df, stats = clusterer.cluster_tickets(df, embeddings=mock_embeddings)

    assert "cluster_id" in res_df.columns
    # Tickets 0 and 1 should share the same cluster_id
    assert res_df.iloc[0]["cluster_id"] == res_df.iloc[1]["cluster_id"]
    # Tickets 2 and 3 should share the same cluster_id
    assert res_df.iloc[2]["cluster_id"] == res_df.iloc[3]["cluster_id"]
    # But they should be distinct clusters
    assert res_df.iloc[0]["cluster_id"] != res_df.iloc[2]["cluster_id"]

    assert stats["total_clusters"] == 2
    assert stats["average_cluster_size"] == 2.0
    assert len(stats["clusters"]) == 2
    assert stats["clusters"][0]["size"] == 2
    assert stats["clusters"][1]["size"] == 2
