# Scientific Evaluation Audit & Benchmark Results

This report documents the verified empirical measurements, confusion matrices, failure classifications, and model comparisons produced by the independent evaluation audit suite.

---

## 1. Retrieval Evaluation Audit

- **Total Test Queries:** 25 (22 Positive, 3 Negative Controls)
- **Precision@5 (Macro):** `0.2636`
- **Recall@5 (Macro):** `0.9924`
- **Mean Reciprocal Rank (MRR):** `1.0000`
- **Mean Average Precision (MAP):** `1.0000`

### Negative Control Resistance (False Positive Shielding):
| Query ID | Dataset | Query | Top Similarity Score | Above 0.60 Threshold | Status |
| --- | --- | --- | --- | --- | --- |
| `Q_GP_06_NEG` | `google_play_reviews` | *cryptocurrency bitcoin payment integration in app* | `0.2572` | `0` | **PASSED (Rejected)** |
| `Q_GH_08_NEG` | `github_issues` | *iOS Swift compilation linker error architecture arm64* | `0.0000` | `0` | **PASSED (Rejected)** |
| `Q_AMZ_04_NEG` | `amazon_product_reviews` | *waterproof IP68 water submersion resistance rating* | `0.0000` | `0` | **PASSED (Rejected)** |

---

## 2. Deduplication Audit & Confusion Matrix

| Dataset | TP | FP | FN | TN | Precision | Recall | Specificity | F1 Score | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `google_play_reviews` | 4 | 0 | 6 | 5 | `1.000` | `0.400` | `1.000` | `0.571` | `0.600` |
| `customer_support_tickets` | 0 | 0 | 0 | 4 | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |
| `amazon_product_reviews` | 0 | 0 | 0 | 3 | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` |

---

## 3. Deduplication Candidate Filters Benchmark

Evaluating whether secondary deterministic filters (Jaccard, N-grams, Topic) improve precision without harming recall:

| Candidate Feature Pipeline | TP | FP | FN | TN | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (Embedding Cosine >= 0.85) | 7 | 0 | 3 | 5 | `1.000` | `0.700` | `0.824` |
| Cosine >= 0.85 + Jaccard >= 0.20 | 5 | 0 | 5 | 5 | `1.000` | `0.500` | `0.667` |
| Cosine >= 0.85 + Jaccard >= 0.30 | 1 | 0 | 9 | 5 | `1.000` | `0.100` | `0.182` |
| Cosine >= 0.85 + 3-gram >= 0.25 | 1 | 0 | 9 | 5 | `1.000` | `0.100` | `0.182` |
| Cosine >= 0.80 + Jaccard >= 0.30 | 1 | 0 | 9 | 5 | `1.000` | `0.100` | `0.182` |
| Cosine >= 0.85 + Exact Topic Match | 3 | 0 | 7 | 5 | `1.000` | `0.300` | `0.462` |

---

## 4. Embedding Generation Performance Audit

### Batch Size Scaling on CPU:
| Batch Size | Sample Size | Elapsed Time | Throughput | Peak RAM | Latency / Text |
| --- | --- | --- | --- | --- | --- |
| `16` | 500 | `6.393s` | **78.2 texts/s** | `0.96 MB` | `12.79 ms` |
| `32` | 500 | `5.496s` | **91.0 texts/s** | `0.96 MB` | `10.99 ms` |
| `64` | 500 | `4.560s` | **109.7 texts/s** | `0.97 MB` | `9.12 ms` |
| `128` | 500 | `2.698s` | **185.3 texts/s** | `0.99 MB` | `5.4 ms` |
| `256` | 500 | `2.522s` | **198.3 texts/s** | `1.03 MB` | `5.04 ms` |

### Determinism & Caching Verification:
- **Strict Numerical Determinism:** `True` (Max $\Delta = 0.0e+00$)
- **All Vectors $L_2$ Normalized:** `True` (Norm Violations: `0`)
- **Cache Speedup Factor:** `397.6x` (`1.4887s` cold vs `0.003744s` cached)

---

## 5. Embedding Model Architecture Comparison

| Model Architecture | Parameters | Disk Size | Throughput (CPU) | Precision@5 | Recall@5 | MRR | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `all-MiniLM-L6-v2` | 22.7M | 90 MB | 533.0 txt/s | `0.264` | `0.992` | `1.000` | **OPTIMAL** |
| `bge-small-en-v1.5` | 33.4M | 133 MB | 131.8 txt/s | `0.264` | `0.992` | `0.977` | **VIABLE_ALTERNATIVE** |
| `e5-small-v2` | 33.4M | 133 MB | 113.4 txt/s | `0.264` | `0.992` | `0.966` | **VIABLE_ALTERNATIVE** |

---

## 6. Threshold Revalidation & 95% Confidence Intervals

### Evidence Similarity Threshold Sweep (Multi-Dataset, Bootstrap N=1000):
| Threshold | Precision | Recall | F1 Score | F1 95% Confidence Interval | Retention Rate |
| --- | --- | --- | --- | --- | --- |
| `0.40` | `0.924` | `0.917` | `0.875` | `[0.760, 0.969]` | `46.9%` |
| `0.50` | `1.000` | `0.826` | `0.831` | `[0.662, 0.955]` | `38.6%` |
| `0.60` | `1.000` | `0.568` | `0.576` | `[0.364, 0.773]` | `27.7%` |
| `0.65` | `1.000` | `0.341` | `0.348` | `[0.167, 0.545]` | `20.8%` |
| `0.70` | `1.000` | `0.159` | `0.167` | `[0.045, 0.334]` | `10.8%` |
| `0.75` | `1.000` | `0.068` | `0.076` | `[0.000, 0.197]` | `5.5%` |
| `0.80` | `1.000` | `0.000` | `0.000` | `[0.000, 0.000]` | `0.0%` |

### Deduplication Threshold Sweep (Confusion Matrix):
| Threshold | TP | FP | FN | TN | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `0.70` | 4 | 0 | 6 | 12 | `1.000` | `0.400` | `0.571` |
| `0.75` | 4 | 0 | 6 | 12 | `1.000` | `0.400` | `0.571` |
| `0.80` | 4 | 0 | 6 | 12 | `1.000` | `0.400` | `0.571` |
| `0.85` | 4 | 0 | 6 | 12 | `1.000` | `0.400` | `0.571` |
| `0.90` | 1 | 0 | 9 | 12 | `1.000` | `0.100` | `0.182` |
| `0.95` | 0 | 0 | 10 | 12 | `1.000` | `0.000` | `0.000` |
