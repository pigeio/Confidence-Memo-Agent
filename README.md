# Confidence Memo Agent

An AI-powered evidence analysis engine that helps product teams make better decisions by telling them honestly what their customer support data does and doesn't support before they build.

---

## 🎯 Core Philosophy

Instead of acting like another overconfident AI product advisor that blindly recommends building every requested feature, **Confidence Memo Agent** operates as an honest evidence analyst. 

The moat is:
- **Calibrated Confidence:** Assigns realistic confidence scores (Low / Moderate / High) based strictly on ticket volume and evidence strength.
- **Evidence Transparency:** Extracts concrete facts, ticket counts, and user quotes from real customer feedback.
- **Honest Uncertainty:** Explicitly calls out missing data (e.g. lack of product analytics, usage metrics, or technical scoping) and recommends further investigation when evidence is weak.

---

## 🏗️ Production Architecture

```text
                               User Feature Proposal + Context
                            (Effort, Impact, Strategy, Cost, Risk)
                                           │
                                           ▼
                                  Semantic Retrieval
                            (Sentence Transformers & NumPy)
                                           │
                                           ▼
                                 Evidence Deduplication
                          (Exact & Dense Embedding Cosine Matrix)
                                           │
                                           ▼
                                  Evidence Clustering
                           (Hierarchical Semantic Clustering)
                                           │
                                           ▼
                                  Evidence Validation
                                           │
                                           ▼
                                Evidence Scoring Engine
                             (Deterministic Multi-Factor)
                                           │
                                           ▼
                              Deterministic Decision Engine
                          (Multi-Criteria Matrix & Gating Logic)
                                           │
                                           ▼
                              Historical Calibration Store
                         (Brier Score, ECE, Reliability Curves)
                                           │
                                           ▼
                               Decision Prompt Builder
                            (Template Engine & Formatting)
                                           │
                                           ▼
                                     Gemini Client
                             (google-genai & Retry Logic)
                                           │
                                           ▼
                                     Memo Service
                               (Pipeline Orchestration)
                                           │
                                           ▼
                                     Decision Memo
                         (Calibrated Executive Decision Report)
```

---

## ✨ Features Completed

### ✅ Sprint 1: Backend Modularization
- **Modular Retrieval Interface (`src/retrieval.py`):** Clean separation for ticket querying with schema validation.
- **Prompt Builder (`src/prompt_builder.py`):** Externalized prompt template engine (`prompts/evidence_prompt.txt`, `prompts/decision_prompt.txt`) with fallback support.
- **Gemini API Client (`src/gemini_client.py`):** Integration with the new `google-genai` SDK featuring exponential backoff retries for transient errors.
- **Memo Service (`src/memo_service.py`):** Unified pipeline orchestration service.

### ✅ Sprint 2: Semantic Retrieval Engine
- **Dense Vector Embeddings (`src/embedding_generator.py`):** Uses `SentenceTransformer('all-MiniLM-L6-v2')` with a model singleton to avoid re-initialization overhead.
- **DataFrame Embedding Cache:** In-memory dataset hash caching (MD5) to avoid re-encoding ticket DataFrames across queries.
- **NumPy Cosine Similarity (`src/similarity_engine.py`):** Fast, zero-FAISS vector dot product search over pre-normalized embeddings.
- **Top-k Relevance Ranking:** Ranks and filters candidates based on configurable similarity thresholds.

### ✅ Sprint 3: Ingestion, Connectors & Multi-Source Pipelines
- **Universal File Ingestion:** CSV, Excel, JSON, TXT, and PDF parser with automatic schema normalization.
- **External Connectors:** Google Sheets, Zendesk, Notion, and Intercom connectors with OAuth/API key authentication.
- **Deterministic Evidence Validation & Scoring:** 5-factor deterministic evidence scoring (Volume, Severity, Sentiment Consistency, Recency, User Diversity).

### ✅ Sprint 3.6: Advanced Evidence Intelligence
- **Evidence Deduplication (`src/evidence_deduplication.py`):** Detects exact text matches (normalization) and semantic duplicate feedback via dense vector similarity matrices with representative ticket selection strategies (`longest`, `highest_similarity`, `earliest`) and rich deduplication telemetry.
- **Evidence Clustering (`src/evidence_clustering.py`):** Groups customer feedback into distinct thematic clusters using agglomerative cosine distance clustering, extracts medoid themes, and calculates cluster distribution metrics.
- **Historical Calibration (`src/historical_calibration.py`):** Storage-abstracted prediction and ground-truth logging (`InMemoryCalibrationStorage`, `JSONCalibrationStorage`) with statistical Brier score, Expected Calibration Error (ECE), reliability diagram curves, and confidence tier accuracy metrics.
- **Deterministic Decision Engine (`src/decision_engine.py`):** Multi-criteria product decision engine combining evidence scores, engineering effort, business impact, strategic alignment, cost, and risk into deterministic recommendations (`PROCEED_TO_BUILD`, `VALIDATE_FURTHER`, `PROTOTYPE_OR_SPIKE`, `DEPRIORITIZE`, `REJECT`), priority scores, assumptions, risks, and trade-offs.
- **Decision Memos (`src/memo_service.py`):** Executive-ready decision memo orchestration combining evidence intelligence with strategic evaluation.

### ✅ Sprint 4: Real-World Dataset Validation & Benchmarking
- **Dataset Adapter Layer (`src/evaluation/adapters.py`):** Pluggable adapters transforming Google Play Reviews, GitHub Issues, Customer Support Tweets, and Amazon Product Reviews into the canonical schema.
- **Config-Driven Dataset Registry (`data/evaluation/datasets.json`):** Dynamic registry for zero-code dataset onboarding.
- **High-Scale Performance Benchmarks (`src/evaluation/benchmark.py`):** Evaluated scaling from 1K to 50K records. Proved pure NumPy cosine retrieval executes in **40 ms** at 50,000 records (220 MB peak RAM).
- **Quantitative Quality Metrics (`src/evaluation/quality.py`):** Evaluated Retrieval (Precision@5: **0.467**, Recall@5: **0.944**, MRR: **1.000**, MAP: **1.000**), Deduplication (F1: **0.571**), and Calibration (Brier: **0.0000** / **0.0420**).
- **Empirical Threshold Optimization (`src/evaluation/threshold_optimizer.py`):** Validated optimal operating points (`EVIDENCE_THRESHOLD = 0.60`, `DEDUP_THRESHOLD = 0.85`, `CLUSTER_DISTANCE = 0.35`).
- **Comprehensive Evaluation Report:** Documented in [`docs/sprint_4_evaluation_report.md`](docs/sprint_4_evaluation_report.md) with example memos in [`evaluation/examples/`](evaluation/examples/).
- **Robustness & Edge-Case Testing (`tests/test_evaluation_edge_cases.py`):** Hardened against missing fields, duplicate IDs, multilingual text, emojis, and 10K+ character payloads.

---

## 🛠️ Installation & Setup

### 1. Clone & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/pigeio/Confidence-Memo-Agent.git
cd Confidence-Memo-Agent

# Create a virtual environment
python -m venv .venv
```

### 2. Activate Virtual Environment

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a `.env` file in the root folder of the project:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

---

## 🧪 Running Tests

The repository features 100% deterministic unit test suites with 94%+ test coverage.

### Run All Unit Tests:
```bash
python -m pytest
```

### Run Tests with Coverage Report:
```bash
python -m pytest --cov=src --cov-report=term-missing
```

---

## 🚀 Running the Pipeline Demo

To run the end-to-end integration demonstration script:

```bash
python tests/run_pipeline_demo.py
```
