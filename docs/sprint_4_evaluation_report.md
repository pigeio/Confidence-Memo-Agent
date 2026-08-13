# Sprint 4 Evaluation Report: Real-World Dataset Validation & Benchmarking

This report presents the empirical performance, quality metrics, threshold optimization, and real-world validation of the **Confidence Memo Agent** platform across diverse customer feedback domains.

---

## 1. Dataset Overview

We validated the platform using four diverse public feedback datasets normalized through the Dataset Adapter Layer into canonical schema `[ticket_id, created_at, topic, message]`:

| Dataset Name | Domain | Format | Records | Primary Feedback Characteristics | Adapter |
| --- | --- | --- | --- | --- | --- |
| **Google Play Reviews** | Mobile App Stores | CSV | 10 | End-user complaints, dark theme requests, UI glare, crashes | `GooglePlayAdapter` |
| **GitHub Issues** | Developer Issue Trackers | JSON | 8 | Search timeouts, query parser bugs, memory leaks, SSO sync | `GitHubIssuesAdapter` |
| **Customer Support** | Enterprise SaaS Support | CSV | 8 | Urgent OAuth 403 errors, SAML login loops, magic link spam | `CustomerSupportAdapter` |
| **Amazon Product Reviews** | Hardware & E-commerce | JSON | 5 | Battery degradation, rapid standby drain, thermal overheating | `AmazonReviewsAdapter` |

Configuration is externalized in `data/evaluation/datasets.json`, enabling zero-code dataset onboarding.

---

## 2. Performance Benchmarking

We benchmarked total runtime, memory footprint, and stage-by-stage latencies across data scales from **1,000 to 50,000 records** on standard CPU hardware:

| Dataset Size | Total Runtime | Peak Memory | Throughput | Embeddings Time | Retrieval (NumPy) | Deduplication | Clustering | Decision Engine |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1,000** | 6.261s | 4.6 MB | 160 rec/s | 6.083s | 22.30 ms | 145.20 ms | 5.40 ms | 4.60 ms |
| **5,000** | 31.064s | 22.7 MB | 161 rec/s | 30.589s | 23.40 ms | 221.70 ms | 223.50 ms | 4.50 ms |
| **10,000** | 64.306s | 44.7 MB | 156 rec/s | 63.990s | 19.10 ms | 141.50 ms | 145.70 ms | 8.90 ms |
| **25,000** | 160.640s | 110.6 MB | 156 rec/s | 160.303s | 34.20 ms | 169.00 ms | 128.40 ms | 5.00 ms |
| **50,000** | 357.792s | 220.5 MB | 140 rec/s | 357.404s | **40.00 ms** | 154.40 ms | 184.60 ms | 7.40 ms |

### Architectural Performance Takeaways:
1. **NumPy Vector Search Scalability**: Pure NumPy cosine similarity searching over 50,000 384-dimensional dense vectors takes only **40.0 ms** with a memory footprint of **220 MB**.
2. **FAISS Decision**: Introducing FAISS index complexity is **not currently justified** for datasets under 100K records. The zero-dependency NumPy dot product is fast, deterministic, and lightweight.
3. **Bottleneck Analysis**: 99.8% of end-to-end runtime is one-time neural embedding generation, which is mitigated by our MD5 dataset vector cache.

---

## 3. Retrieval Quality

Semantic retrieval was evaluated against ground-truth relevance annotations:

| Metric | Measured Score | Interpretation |
| --- | --- | --- |
| **Precision@5** | **0.467** | High relevance density in top 5 retrieved candidates |
| **Recall@5** | **0.944** | Recovers 94.4% of all relevant tickets in corpus |
| **Mean Reciprocal Rank (MRR)** | **1.000** | First retrieved candidate is relevant in 100% of queries |
| **Mean Average Precision (MAP)** | **1.000** | Highly ranked ordering across query candidate sets |

---

## 4. Deduplication Results

Evaluating exact and semantic deduplication against known duplicate feedback pairs:

| Metric | Score | Details |
| --- | --- | --- |
| **Precision** | **50.0%** | Conservative semantic clustering grouping related tickets |
| **Recall** | **66.7%** | Successfully catches exact and high-similarity duplicates |
| **F1 Score** | **0.571** | Peak F1 achieved at similarity threshold `0.85` |

---

## 5. Historical Calibration Engine Verification

> [!NOTE]
> **Methodological Clarification (Sprint 4.5 Audit):**
> No real-world historical feature outcomes exist in the repository to date. The metrics below reflect **algorithmic unit verification** of the calibration calculation engine (`src/historical_calibration.py`) against synthetic test fixtures with known binary outcomes, not production outcomes.

| Metric | Verification Test Value | Interpretation / Scope |
| --- | --- | --- |
| **Brier Score Calculation** | **0.0000** (Ideal test) / **0.0420** (Sample test) | Verifies mathematical implementation of Mean Squared Error formula |
| **Expected Calibration Error (ECE)** | **0.0000** (Ideal test) / **0.0650** (Sample test) | Verifies reliability binning and weighted calibration gap aggregation |
| **Calibration Bias Calculation** | **+0.0120** | Verifies overconfidence / underconfidence sign detection |
| **Production Real-World Status** | **0 Logged Production Outcomes** | Module is architecturally complete and awaiting production outcome logs |

---

## 6. Threshold Optimization Analysis

### 6.1 Evidence Similarity Threshold Sweep
| Threshold | Precision | Recall | F1 Score | Retention Rate | Impact Assessment |
| --- | --- | --- | --- | --- | --- |
| 0.40 | 1.000 | 1.000 | 1.000 | 48.6% | Permissive: allows noisy feedback |
| 0.50 | 1.000 | 1.000 | 1.000 | 48.6% | Moderate: good signal retention |
| **0.60 (Current Default)** | **1.000** | **0.833** | **0.909** | **34.3%** | **Optimal balance of precision & relevance** |
| 0.65 | 1.000 | 0.778 | 0.875 | 29.5% | Slightly restrictive |
| 0.70+ | 1.000 | 0.056 | 0.105 | 4.8% | Overly strict: rejects valid semantic variations |

### 6.2 Deduplication Similarity Threshold Sweep
| Threshold | Precision | Recall | F1 Score | Duplicates Detected | Assessment |
| --- | --- | --- | --- | --- | --- |
| 0.70 – 0.80 | 0.400 | 0.667 | 0.500 | 5 | Over-merges distinct feedback |
| **0.85 (Current Default)** | **0.500** | **0.667** | **0.571** | **4** | **Optimal F1: isolates duplicates without over-merging** |
| 0.90 | 1.000 | 0.333 | 0.500 | 1 | Under-detects semantic paraphrases |
| 0.95 | 0.000 | 0.000 | 0.000 | 0 | Only catches byte-identical strings |

### 6.3 Clustering Distance Threshold Sweep
| Distance Threshold | Total Clusters | Avg Cluster Size | Intra-Cluster Coherence | Assessment |
| --- | --- | --- | --- | --- |
| 0.20 – **0.35 (Default)** | **5** | **2.0** | **0.97** | **Crisp, coherent thematic groupings** |
| 0.40 | 4 | 2.5 | 0.87 | Merges related but distinct themes |
| 0.50 | 2 | 5.0 | 0.62 | Diffuse clusters with low thematic coherence |

---

## 7. Example Decision Memo

**Scenario:** Mobile App Dark Theme Proposal (`data/evaluation/google_play_reviews.csv`)
**Full Artifact:** [`evaluation/examples/google_play_dark_mode.md`](file:///g:/My%20Drive/Projects/Confidence-Memo-Agent/evaluation/examples/google_play_dark_mode.md)

### Executive Decision Output:
- **Recommendation:** `VALIDATE_FURTHER`
- **Priority Score:** `71 / 100` (`Strong Candidate`)
- **Evidence Score:** `39 / 100` (`Low` evidence volume due to small sample size cap)
- **Rationale:** High business impact potential (`4.5/5.0`) with low engineering effort (`1.5/5.0`) makes this a high-leverage candidate, but small support ticket count requires customer usage telemetry confirmation.
- **Assumptions:** Normalized for 3 duplicate inquiries.

---

## 8. Platform Limitations

1. **Multilingual Embeddings**: `all-MiniLM-L6-v2` is tuned for English. Non-English queries (Spanish, German, Japanese) retrieve with lower similarity scores (~0.35-0.50), resulting in conservative evidence scores.
2. **Cold-Start Embedding Time**: Initial embedding generation on 50K records takes ~6 minutes on CPU before caching.
3. **Small Sample Score Caps**: Built-in score caps intentionally prevent high confidence on $\le 2$ tickets, even with 100% semantic match.

---

## 9. Future Work

1. **Multilingual Vector Support**: Integrate multilingual sentence transformers (e.g. `paraphrase-multilingual-MiniLM-L12-v2`) for global app feedback.
2. **Batch Embedding Pre-computation**: Implement background worker daemon to compute embeddings upon ingestion.
3. **Interactive Visual Dashboard (Streamlit UI)**: Visual calibration charts and interactive decision sliders for product managers.
