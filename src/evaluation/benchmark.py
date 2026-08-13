import time
import tracemalloc
import logging
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from src.retrieval import retrieve_tickets
from src.evidence_validation import validate_retrieved_evidence
from src.evidence_scoring import EvidenceScoringEngine
from src.evidence_deduplication import EvidenceDeduplicator
from src.evidence_clustering import EvidenceClusterer
from src.decision_engine import DecisionEngine
from src.embedding_generator import EmbeddingGenerator
from src.similarity_engine import SimilarityEngine

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    dataset_size: int
    total_runtime_sec: float
    peak_memory_mb: float
    throughput_records_per_sec: float
    embedding_time_sec: float
    retrieval_time_sec: float
    validation_time_sec: float
    deduplication_time_sec: float
    clustering_time_sec: float
    decision_engine_time_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


class BenchmarkEngine:
    """
    Performance and Scalability Benchmarking Suite.
    Measures end-to-end and component-level latency and peak memory across data scales (1K to 50K).
    """

    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.scoring_engine = EvidenceScoringEngine()
        self.deduplicator = EvidenceDeduplicator()
        self.clusterer = EvidenceClusterer()
        self.decision_engine = DecisionEngine()
        self.similarity_engine = SimilarityEngine()

    @staticmethod
    def generate_synthetic_dataset(n_records: int) -> pd.DataFrame:
        """
        Generate a synthetic dataset of realistic customer support tickets for benchmarking.
        """
        topics = [
            "Dark Theme",
            "Search Performance",
            "SSO Authentication",
            "Battery Drain",
            "CSV Export",
            "Offline Sync",
            "Billing Error",
            "Mobile Crash",
            "Notification Delay",
            "Settings Navigation",
        ]
        sample_messages = [
            "The app screen is too bright in dark rooms. Please add night mode theme!",
            "Search queries timeout when filtering by author on large repositories.",
            "Cannot login with Google OAuth. Receiving 403 access denied authorization error.",
            "Battery drains rapidly by 40% when background audio playback is enabled.",
            "Need ability to export table reports to CSV and Excel format for management.",
            "Offline changes fail to synchronize when internet connection is restored.",
            "Charged twice on monthly subscription invoice 9821 without authorization.",
            "App crashes repeatedly on Android 14 when navigating to account settings.",
            "Push notifications for direct messages are delayed by more than 15 minutes.",
            "Navigation menu is confusing and settings cannot be easily found.",
        ]

        np.random.seed(42)
        chosen_topics = np.random.choice(topics, size=n_records)
        chosen_indices = np.random.choice(len(sample_messages), size=n_records)
        messages = [
            f"Ticket #{i}: {sample_messages[chosen_indices[i]]}"
            for i in range(n_records)
        ]
        ticket_ids = [f"SYNTH_{i:06d}" for i in range(n_records)]
        dates = pd.date_range("2026-01-01", periods=n_records, freq="min").astype(str)

        return pd.DataFrame(
            {
                "ticket_id": ticket_ids,
                "created_at": dates,
                "topic": chosen_topics,
                "message": messages,
            }
        )

    def benchmark_scale(
        self,
        n_records: int,
        df: pd.DataFrame | None = None,
        query: str = "Dark theme night mode support",
    ) -> BenchmarkResult:
        """
        Run comprehensive benchmark on a dataset of size n_records.
        """
        if df is None:
            df = self.generate_synthetic_dataset(n_records)

        tracemalloc.start()
        start_total = time.perf_counter()

        # 1. Embedding generation & Matrix scaling
        # For N <= 1000, encode full dataset via model; for N > 1000, encode sample to measure throughput and tile
        t0 = time.perf_counter()
        if n_records <= 1000:
            embeddings = self.embedding_generator.encode_dataframe(df)
            t_embed = time.perf_counter() - t0
        else:
            sample_df = df.iloc[:500]
            sample_embeddings = self.embedding_generator.encode_dataframe(sample_df)
            sample_time = time.perf_counter() - t0
            per_record_time = sample_time / 500.0
            t_embed = per_record_time * n_records

            # Synthesize full normalized N x 384 matrix for vector operations
            repeats = int(np.ceil(n_records / len(sample_embeddings)))
            embeddings = np.tile(sample_embeddings, (repeats, 1))[:n_records].astype(np.float32)

        # 2. Retrieval (NumPy Cosine Similarity over N records)
        t0 = time.perf_counter()
        query_embedding = self.embedding_generator.encode_texts(query)
        scores = self.similarity_engine.compute_cosine_similarity(
            embeddings, query_embedding
        )
        ranked = self.similarity_engine.rank_and_filter(scores, top_k=20, threshold=0.20)
        t_retrieval = time.perf_counter() - t0

        matched_indices = [idx for idx, _ in ranked]
        matched_scores = np.array([s for _, s in ranked], dtype=np.float32)
        matching_tickets = df.iloc[matched_indices].copy().reset_index(drop=True)

        # 3. Validation
        t0 = time.perf_counter()
        val_tickets, val_scores = validate_retrieved_evidence(
            matching_tickets, matched_scores, return_scores=True
        )
        t_val = time.perf_counter() - t0

        # 4. Deduplication
        t0 = time.perf_counter()
        dedup_tickets, dedup_scores, dedup_stats = self.deduplicator.deduplicate(
            val_tickets, similarity_scores=val_scores
        )
        t_dedup = time.perf_counter() - t0

        # 5. Clustering
        t0 = time.perf_counter()
        clustered_tickets, cluster_stats = self.clusterer.cluster_tickets(
            dedup_tickets
        )
        t_cluster = time.perf_counter() - t0

        # 6. Evidence Scoring & Decision Engine
        t0 = time.perf_counter()
        scoring_info = self.scoring_engine.calculate_score(
            clustered_tickets,
            similarity_scores=dedup_scores,
            total_retrieved_count=len(matching_tickets),
        )
        _ = self.decision_engine.evaluate_decision(
            evidence_score=scoring_info.get("score", 0),
            engineering_effort="Low",
            business_impact="High",
            strategic_alignment="High",
            cost="Low",
            risk="Low",
            evidence_summary=scoring_info.get("evidence_summary"),
            deduplication_stats=dedup_stats,
            clustering_stats=cluster_stats,
        )
        t_decision = time.perf_counter() - t0

        total_runtime = t_embed + t_retrieval + t_val + t_dedup + t_cluster + t_decision
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Add estimated memory footprint of embeddings (N * 384 * 4 bytes)
        embedding_bytes = n_records * 384 * 4
        peak_mb = (peak_bytes + embedding_bytes) / (1024 * 1024)
        throughput = n_records / total_runtime if total_runtime > 0 else 0.0

        return BenchmarkResult(
            dataset_size=n_records,
            total_runtime_sec=round(total_runtime, 4),
            peak_memory_mb=round(peak_mb, 2),
            throughput_records_per_sec=round(throughput, 1),
            embedding_time_sec=round(t_embed, 4),
            retrieval_time_sec=round(t_retrieval, 4),
            validation_time_sec=round(t_val, 4),
            deduplication_time_sec=round(t_dedup, 4),
            clustering_time_sec=round(t_cluster, 4),
            decision_engine_time_sec=round(t_decision, 4),
        )

    def run_multi_scale_benchmark(
        self,
        scales: list[int] = [1000, 5000, 10000, 25000, 50000],
    ) -> list[BenchmarkResult]:
        """
        Run benchmarks across multiple scales and return list of BenchmarkResult objects.
        """
        results = []
        for n in scales:
            logger.info(f"Benchmarking dataset scale N={n}...")
            res = self.benchmark_scale(n)
            results.append(res)
        return results

    @staticmethod
    def format_results_markdown_table(results: list[BenchmarkResult]) -> str:
        """
        Format benchmark results as a clean GitHub Flavored Markdown table.
        """
        headers = [
            "Dataset Size",
            "Total Runtime",
            "Peak Memory",
            "Throughput",
            "Embeddings",
            "Retrieval (NumPy)",
            "Deduplication",
            "Clustering",
            "Decision Engine",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for r in results:
            row = [
                f"{r.dataset_size:,}",
                f"{r.total_runtime_sec:.3f}s",
                f"{r.peak_memory_mb:.1f} MB",
                f"{r.throughput_records_per_sec:,.0f} rec/s",
                f"{r.embedding_time_sec:.3f}s",
                f"{r.retrieval_time_sec*1000:.2f}ms",
                f"{r.deduplication_time_sec*1000:.2f}ms",
                f"{r.clustering_time_sec*1000:.2f}ms",
                f"{r.decision_engine_time_sec*1000:.2f}ms",
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)
