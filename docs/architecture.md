# Production Architecture Documentation

This document describes the module architecture of the **Confidence Memo Agent**, detailing the purpose, inputs, outputs, responsibilities, and dependencies of each component in the system.

---

## 📐 High-Level Architecture Diagram

```text
                        ┌────────────────────────────────────────┐
                        │         User Feature Proposal          │
                        │ (Effort, Impact, Strategy, Cost, Risk) │
                        └───────────────────┬────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Ingestion & Connector Layer (src/)                              │
│                                                                                        │
│   preprocessing/ (CSV, Excel, JSON, TXT, PDF parser + normalizer)                      │
│   connectors/    (Google Sheets, Zendesk, Notion, Intercom APIs)                       │
│   evaluation/    (Google Play, GitHub, Customer Support, Amazon Adapters & Registry)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Standard DataFrame [ticket_id, created_at, topic, message]
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Retrieval & Intelligence Layer (src/)                           │
│                                                                                        │
│   retrieval.py               (Public Wrapper)                                          │
│   semantic_search.py         (Vector Dot Product Cosine Search)                        │
│   embedding_generator.py     (Singleton SentenceTransformer & MD5 Cache)               │
│   evidence_validation.py     (Similarity Threshold Gating)                             │
│   evidence_deduplication.py  (Exact Text Normalization & Dense Cosine Matrix)          │
│   evidence_clustering.py     (Hierarchical Semantic Clustering & Medoids)              │
│   evidence_scoring.py        (Deterministic 5-Factor Score & Caps)                     │
│   decision_engine.py         (Multi-Criteria Matrix, Gating & Trade-Offs)               │
│   historical_calibration.py  (Brier Score, ECE & Abstract Storage)                     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Structured Decision & Evidence Telemetry
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                   Prompt Builder Layer (src/ & prompts/)                               │
│                                                                                        │
│   prompt_builder.py  <──> prompts/evidence_prompt.txt, prompts/decision_prompt.txt     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Fully Substituted Prompt String
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      Gemini Client Layer (src/)                                        │
│                                                                                        │
│   gemini_client.py   <──> Google Gemini API (gemini-3.5-flash)                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ API Text Response (Explanation Only)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer (src/)                                         │
│                                                                                        │
│   memo_service.py    ───> Returns Evidence Memo / Decision Memo Markdown String        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Specifications

### 1. `config.py`
- **Purpose:** Centralized configuration management and model default parameters.
- **Inputs:** Environment variables (`EMBEDDING_MODEL_NAME`, `DEFAULT_TOP_K`, `DEFAULT_SIMILARITY_THRESHOLD`, `EVIDENCE_SIMILARITY_THRESHOLD`, `DEDUPLICATION_SIMILARITY_THRESHOLD`, `CLUSTERING_DISTANCE_THRESHOLD`).
- **Outputs:** Module-level constants across the codebase.

### 2. `embedding_generator.py`
- **Purpose:** Vector embedding generation, neural model lifecycle management, and ticket dataset caching.
- **Inputs:** Raw text strings or pandas DataFrame (`topic`, `message` columns).
- **Outputs:** 2D NumPy array of L2-normalized dense embeddings (`(N, 384)`).
- **Responsibilities:**
  - Maintain a singleton instance of `SentenceTransformer('all-MiniLM-L6-v2')`.
  - In-memory dataset MD5 hash caching to avoid re-encoding.

### 3. `similarity_engine.py`
- **Purpose:** Pure NumPy vector math for computing cosine similarity and candidate ranking.
- **Inputs:** Ticket embedding matrix (`N, D`), query vector (`1, D`), `top_k`, `threshold`.
- **Outputs:** List of `(index, similarity_score)` tuples sorted in descending order of relevance.

### 4. `evidence_validation.py`
- **Purpose:** Evidence relevance gating against `EVIDENCE_SIMILARITY_THRESHOLD` (0.60).
- **Responsibilities:** Pure filtering module that excludes tickets scoring below threshold.

### 5. `evidence_deduplication.py`
- **Purpose:** Exact text normalization and dense vector semantic deduplication.
- **Responsibilities:** Detects duplicate inquiries, applies representative selection strategies (`longest`, `highest_similarity`, `earliest`), and provides duplicate telemetry.

### 6. `evidence_clustering.py`
- **Purpose:** Hierarchical semantic feedback clustering and medoid theme extraction.
- **Responsibilities:** Groups tickets using cosine distance clustering, extracts theme labels, and calculates intra-cluster coherence.

### 7. `evidence_scoring.py`
- **Purpose:** 5-factor deterministic evidence scoring engine (Volume, Severity, Sentiment Consistency, Recency, User Diversity).
- **Responsibilities:** Clamps scores $0-100$ and applies single/two-ticket score caps.

### 8. `decision_engine.py`
- **Purpose:** Multi-criteria product decision engine.
- **Inputs:** Evidence score, engineering effort, business impact, strategic alignment, cost, risk.
- **Outputs:** Deterministic recommendation (`PROCEED_TO_BUILD`, `VALIDATE_FURTHER`, `PROTOTYPE_OR_SPIKE`, `DEPRIORITIZE`, `REJECT`), priority score, assumptions, risks, trade-offs.

### 9. `historical_calibration.py`
- **Purpose:** Storage-abstracted historical outcome logging and statistical calibration metrics.
- **Responsibilities:** Calculates Brier score, Expected Calibration Error (ECE), reliability diagrams, and calibration drift.

### 10. `evaluation/` Package
- **`adapters.py`**: Domain adapters for Google Play, GitHub Issues, Customer Support, and Amazon Reviews.
- **`benchmark.py`**: Multi-scale latency & memory benchmarking engine.
- **`quality.py`**: Evaluation metrics suite (Precision@k, Recall@k, MRR, MAP, F1, Silhouette, ECE).
- **`threshold_optimizer.py`**: Empirical grid sweep and sensitivity analysis.
- **`e2e_validator.py`**: Multi-domain pipeline validation and example generator.

### 11. `prompt_builder.py`
- **Purpose:** Formats tickets, scoring, telemetry, decision outputs, assumptions, and risks into strict prompt templates.

### 12. `gemini_client.py`
- **Purpose:** Robust Google Gemini API client with exponential backoff retries for transient errors.

### 13. `memo_service.py`
- **Purpose:** Unified orchestration layer for Evidence Memos and Decision Memos.
