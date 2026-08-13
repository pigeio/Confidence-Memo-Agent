import os
import json
import pandas as pd
from src.evaluation.adapters import DatasetRegistry
from src.evaluation.benchmark import BenchmarkEngine
from src.evaluation.quality import QualityEvaluator
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from src.evaluation.e2e_validator import E2EValidator


def main():
    print("==================================================")
    print("=== SPRINT 4: REAL-WORLD VALIDATION & BENCHMARKING ===")
    print("==================================================\n")

    # Step 1: Run End-to-End Scenarios & Generate Examples
    print("[1/4] Running End-to-End Scenarios on Real-World Datasets...")
    validator = E2EValidator()
    e2e_results = validator.run_all_scenarios(output_dir="evaluation/examples", mock_llm=True)
    for r in e2e_results:
        print(f"  * [{r['dataset']}] '{r['proposal']}' -> {r['recommendation']} (Score: {r['evidence_score']}, Priority: {r['priority_score']})")

    # Step 2: Run Multi-Scale Performance Benchmarks
    print("\n[2/4] Running Multi-Scale Performance Benchmarking (1K - 50K records)...")
    benchmark_engine = BenchmarkEngine()
    scales = [1000, 5000, 10000, 25000, 50000]
    bench_results = benchmark_engine.run_multi_scale_benchmark(scales=scales)
    table_md = BenchmarkEngine.format_results_markdown_table(bench_results)
    print("\nBenchmark Results Table:")
    print(table_md)

    # Step 3: Run Quality Evaluations
    print("\n[3/4] Running Quality Metrics on Evaluation Datasets...")
    # 3.1 Retrieval Quality
    gp_df, _ = DatasetRegistry.load_dataset("google_play_reviews")
    gp_queries = [
        {"query": "dark mode night theme", "relevant_ticket_ids": {"gp_1001", "gp_1002", "gp_1003", "gp_1004", "gp_1009", "gp_1010"}},
        {"query": "search broken playlist", "relevant_ticket_ids": {"gp_1005"}},
        {"query": "app crashes settings", "relevant_ticket_ids": {"gp_1007"}},
    ]
    retrieval_metrics = QualityEvaluator.evaluate_retrieval(gp_queries, gp_df, k=5)
    print(f"  Retrieval Quality (Play Store): Precision@5 = {retrieval_metrics['precision_at_k']:.3f}, Recall@5 = {retrieval_metrics['recall_at_k']:.3f}, MRR = {retrieval_metrics['mrr']:.3f}, MAP = {retrieval_metrics['map']:.3f}")

    # 3.2 Deduplication Quality
    gp_gt_dups = {("gp_1001", "gp_1002"), ("gp_1001", "gp_1004"), ("gp_1002", "gp_1004")}
    dedup_metrics = QualityEvaluator.evaluate_deduplication(gp_df, gp_gt_dups, similarity_threshold=0.85)
    print(f"  Deduplication Quality: Precision = {dedup_metrics['precision']:.3f}, Recall = {dedup_metrics['recall']:.3f}, F1 = {dedup_metrics['f1_score']:.3f}")

    # Step 4: Run Threshold Optimization Sweeps
    print("\n[4/4] Running Threshold Optimization Sweeps...")
    ev_sweep = ThresholdOptimizer.sweep_evidence_similarity_threshold(gp_df, gp_queries, thresholds=[0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80])
    dedup_sweep = ThresholdOptimizer.sweep_deduplication_threshold(gp_df, gp_gt_dups, thresholds=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    cluster_sweep = ThresholdOptimizer.sweep_clustering_distance_threshold(gp_df, distance_thresholds=[0.20, 0.25, 0.30, 0.35, 0.40, 0.50])

    print("  Evidence Threshold Sweep:")
    for s in ev_sweep:
        print(f"    Thresh {s['threshold']:.2f} -> P: {s['precision']:.2f}, R: {s['recall']:.2f}, F1: {s['f1_score']:.2f}, Retention: {s['average_retention_rate']*100:.1f}%")

    print("  Deduplication Threshold Sweep:")
    for s in dedup_sweep:
        print(f"    Thresh {s['threshold']:.2f} -> P: {s['precision']:.2f}, R: {s['recall']:.2f}, F1: {s['f1_score']:.2f}, Detected: {s['duplicates_detected']}")

    print("  Clustering Distance Sweep:")
    for s in cluster_sweep:
        print(f"    Dist {s['distance_threshold']:.2f} -> Clusters: {s['total_clusters']}, Avg Size: {s['average_cluster_size']:.1f}, Coherence: {s['intra_cluster_coherence']:.2f}")

    # Output JSON bundle for report integration
    summary_data = {
        "benchmarks": [r.to_dict() for r in bench_results],
        "retrieval_quality": retrieval_metrics,
        "deduplication_quality": dedup_metrics,
        "evidence_threshold_sweep": ev_sweep,
        "deduplication_threshold_sweep": dedup_sweep,
        "clustering_threshold_sweep": cluster_sweep,
    }
    with open("data/evaluation/sprint_4_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print("\nSaved evaluation metrics bundle to data/evaluation/sprint_4_metrics.json")


if __name__ == "__main__":
    main()
