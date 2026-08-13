import pytest
import pandas as pd
from src.evaluation.benchmark import BenchmarkEngine, BenchmarkResult


def test_generate_synthetic_dataset():
    df = BenchmarkEngine.generate_synthetic_dataset(100)
    assert len(df) == 100
    assert set(df.columns) == {"ticket_id", "created_at", "topic", "message"}
    assert df["ticket_id"].nunique() == 100


def test_benchmark_scale_small():
    engine = BenchmarkEngine()
    df = engine.generate_synthetic_dataset(50)
    result = engine.benchmark_scale(n_records=50, df=df)

    assert isinstance(result, BenchmarkResult)
    assert result.dataset_size == 50
    assert result.total_runtime_sec > 0.0
    assert result.peak_memory_mb >= 0.0
    assert result.embedding_time_sec >= 0.0
    assert result.retrieval_time_sec >= 0.0
    assert result.deduplication_time_sec >= 0.0
    assert result.clustering_time_sec >= 0.0
    assert result.decision_engine_time_sec >= 0.0


def test_format_results_markdown_table():
    r1 = BenchmarkResult(
        dataset_size=1000,
        total_runtime_sec=2.5,
        peak_memory_mb=120.0,
        throughput_records_per_sec=400.0,
        embedding_time_sec=2.4,
        retrieval_time_sec=0.005,
        validation_time_sec=0.001,
        deduplication_time_sec=0.002,
        clustering_time_sec=0.003,
        decision_engine_time_sec=0.001,
    )
    table_str = BenchmarkEngine.format_results_markdown_table([r1])
    assert "| Dataset Size |" in table_str
    assert "1,000" in table_str
    assert "2.500s" in table_str
