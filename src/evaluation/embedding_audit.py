import time
import tracemalloc
import logging
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from src.embedding_generator import (
    EmbeddingGenerator,
    get_embedding_model,
    clear_embedding_cache,
    get_embedding_cache,
    compute_dataset_hash,
)
from src.evaluation.benchmark import BenchmarkEngine

logger = logging.getLogger(__name__)


class EmbeddingAuditor:
    """
    Embedding Generation & Hardware Performance Audit Suite.
    Measures batch sizes, throughput, caching efficiency, vector determinism, and memory usage.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = get_embedding_model(model_name)

    def benchmark_batch_sizes(
        self,
        batch_sizes: list[int] = [16, 32, 64, 128, 256],
        sample_size: int = 500,
    ) -> list[dict]:
        """
        Benchmark sentence encoding across different batch sizes to determine optimal CPU throughput.
        """
        synth_df = BenchmarkEngine.generate_synthetic_dataset(sample_size)
        texts = [
            f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
            for _, row in synth_df.iterrows()
        ]

        results = []
        for bs in batch_sizes:
            tracemalloc.start()
            t0 = time.perf_counter()
            embeddings = self.model.encode(
                texts,
                batch_size=bs,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            elapsed = time.perf_counter() - t0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            throughput = len(texts) / elapsed if elapsed > 0 else 0.0
            peak_mb = peak_bytes / (1024 * 1024)

            results.append(
                {
                    "batch_size": bs,
                    "sample_size": sample_size,
                    "elapsed_seconds": round(elapsed, 4),
                    "throughput_texts_per_sec": round(throughput, 1),
                    "peak_memory_mb": round(peak_mb, 2),
                    "avg_ms_per_text": round((elapsed / len(texts)) * 1000, 2),
                }
            )

        return results

    def verify_determinism_and_norm(
        self,
        sample_size: int = 50,
    ) -> dict:
        """
        Verify that embedding vectors are strictly deterministic across runs and properly L2-normalized.
        """
        synth_df = BenchmarkEngine.generate_synthetic_dataset(sample_size)
        texts = [
            f"Topic: {row.get('topic', '')} | Message: {row.get('message', '')}"
            for _, row in synth_df.iterrows()
        ]

        emb1 = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        emb2 = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

        # 1. Exact numeric equivalence
        is_exact = np.array_equal(emb1, emb2)
        max_diff = float(np.max(np.abs(emb1 - emb2)))

        # 2. L2 Unit Norm check
        norms = np.linalg.norm(emb1, axis=1)
        norm_violations = np.sum(np.abs(norms - 1.0) > 1e-5)

        return {
            "is_deterministic": bool(is_exact),
            "max_absolute_difference": max_diff,
            "all_l2_unit_normalized": bool(norm_violations == 0),
            "norm_violation_count": int(norm_violations),
            "min_norm": round(float(np.min(norms)), 6),
            "max_norm": round(float(np.max(norms)), 6),
        }

    def verify_caching_performance(
        self,
        sample_size: int = 200,
    ) -> dict:
        """
        Verify that DataFrame hash caching eliminates redundant re-computations completely.
        """
        clear_embedding_cache()
        generator = EmbeddingGenerator(model_name=self.model_name)
        df = BenchmarkEngine.generate_synthetic_dataset(sample_size)

        # Run 1: Cold start
        t0 = time.perf_counter()
        emb_cold = generator.encode_dataframe(df)
        cold_time = time.perf_counter() - t0

        # Run 2: Cached hit
        t0 = time.perf_counter()
        emb_cached = generator.encode_dataframe(df)
        cached_time = time.perf_counter() - t0

        speedup = cold_time / cached_time if cached_time > 0 else 0.0

        return {
            "sample_size": sample_size,
            "cold_start_seconds": round(cold_time, 4),
            "cached_lookup_seconds": round(cached_time, 6),
            "speedup_factor": round(speedup, 1),
            "cache_hit_verified": np.array_equal(emb_cold, emb_cached),
            "cache_entries_count": len(get_embedding_cache()),
        }
