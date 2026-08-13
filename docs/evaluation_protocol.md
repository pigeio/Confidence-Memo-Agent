# Formal Evaluation Protocol — Confidence Memo Agent

This document defines the scientific evaluation protocol, mathematical metrics, dataset annotation standards, reproducibility controls, and benchmarking methodologies for the **Confidence Memo Agent** platform.

---

## 1. Overview & Evaluation Principles

1. **Deterministic Separation:** All confidence scoring, evidence retrieval, deduplication, clustering, and decision recommendations are deterministic Python modules. Gemini only generates explanatory prose.
2. **Zero Verbatim Leakage:** Retrieval evaluation queries must never copy exact phrasing from target documents.
3. **Negative Control Inclusion:** Evaluation datasets must include negative control queries with zero matching documents in corpus to measure false-positive rejection.
4. **Complete Confusion Matrix Reporting:** Classification and deduplication tasks must report full True Positive, False Positive, False Negative, and True Negative counts.

---

## 2. Dataset Schema & Domain Sources

All datasets are adapted into the canonical project schema:
$$\text{Schema} = [\texttt{ticket\_id: str}, \texttt{created\_at: str}, \texttt{topic: str}, \texttt{message: str}]$$

| Dataset Key | Source / Domain | Format | Ground Truth Reference File |
| --- | --- | --- | --- |
| `google_play_reviews` | Mobile App Store Customer Reviews | CSV | `data/evaluation/google_play_reviews.csv` |
| `github_issues` | Developer Open Source Issue Tracker | JSON | `data/evaluation/github_issues.json` |
| `customer_support_tickets` | Enterprise SaaS Auth & Support Tickets | CSV | `data/evaluation/customer_support_tickets.csv` |
| `amazon_product_reviews` | Hardware E-commerce Reviews | JSON | `data/evaluation/amazon_product_reviews.json` |

Registry configuration: `data/evaluation/datasets.json`.

---

## 3. Metric Mathematical Definitions

### 3.1 Retrieval Metrics
Let $Q$ be the set of evaluation queries. For query $q \in Q$, let $\text{Rel}(q)$ be the set of ground-truth relevant document IDs, and let $R_k(q) = [d_1, d_2, \dots, d_k]$ be the top-$k$ retrieved documents ordered by cosine similarity score.

- **Precision@k:**
  $$\text{Precision@}k(q) = \frac{|\{d \in R_k(q) \mid d \in \text{Rel}(q)\}|}{k}$$

- **Recall@k:**
  $$\text{Recall@}k(q) = \frac{|\{d \in R_k(q) \mid d \in \text{Rel}(q)\}|}{|\text{Rel}(q)|} \quad (\text{if } |\text{Rel}(q)| > 0)$$

- **Reciprocal Rank (RR):**
  $$\text{RR}(q) = \frac{1}{\min \{ i \mid d_i \in \text{Rel}(q) \}} \quad (\text{or } 0 \text{ if no relevant item retrieved})$$
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \text{RR}(q)$$

- **Average Precision (AP@k):**
  $$\text{AP@}k(q) = \frac{1}{\min(k, |\text{Rel}(q)|)} \sum_{i=1}^k \text{Precision@}i(q) \cdot \mathbb{I}(d_i \in \text{Rel}(q))$$
  $$\text{MAP} = \frac{1}{|Q|} \sum_{q \in Q} \text{AP@}k(q)$$

### 3.2 Deduplication Metrics
Let $\mathcal{P}$ be the set of all evaluated pairs $(d_i, d_j)$.
- $\text{TP} = |\{ (d_i, d_j) \in \mathcal{P} \mid \text{Predicted}=1 \land \text{GroundTruth}=1 \}|$
- $\text{FP} = |\{ (d_i, d_j) \in \mathcal{P} \mid \text{Predicted}=1 \land \text{GroundTruth}=0 \}|$
- $\text{FN} = |\{ (d_i, d_j) \in \mathcal{P} \mid \text{Predicted}=0 \land \text{GroundTruth}=1 \}|$
- $\text{TN} = |\{ (d_i, d_j) \in \mathcal{P} \mid \text{Predicted}=0 \land \text{GroundTruth}=0 \}|$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3.3 Historical Calibration Metrics
Let $p_i \in [0, 1]$ be the predicted confidence and $y_i \in \{0, 1\}$ be the actual outcome for record $i \in \{1, \dots, N\}$:
- **Brier Score:**
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
- **Expected Calibration Error (ECE):**
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 4. Reproducibility & Random Seed Protocol

To reproduce all benchmarks, evaluations, and reports identically:
1. **Random Seed:** Set `seed = 42` for all `numpy` and `random` generators.
2. **Model Version:** Use `sentence-transformers` `all-MiniLM-L6-v2` (MD5 checksum verified).
3. **Execution Command:**
   ```bash
   python -m src.evaluation.run_evaluation
   ```
4. **Outputs Generated:**
   - Metrics JSON: `data/evaluation/audit_metrics.json`
   - Evaluation Results Report: `docs/audit/evaluation_results.md`

---

## 5. Known Limitations & Scope

1. **Domain Diversity:** Current evaluation covers 4 diverse public domains (App Store, GitHub, Support, E-Commerce), but enterprise ticket volumes ($>100\text{k}$) require clustered compute.
2. **Historical Outcomes:** Production calibration requires logging $>100$ real feature outcomes after post-launch monitoring.
