# Development Log — Confidence Memo Agent

This document maintains a chronological record of milestones, refactoring phases, and architectural updates accomplished throughout the development of the Confidence Memo Agent.

---

## 📅 Chronological Milestones

### 🟢 Sprint 0: Initial Proof-of-Concept Prototype
- **Environment:** Google Colab notebook.
- **Dataset:** Created initial synthetic dataset of 20 customer support tickets.
- **Implementation:** Monolithic `get_confidence_memo(...)` function containing keyword search, hardcoded prompt strings, and Gemini API calls.
- **Validation:** Demonstrated that LLMs can generate calibrated confidence levels (High, Moderate, Weak/Low) based on evidence.

---

### 🟢 Sprint 1: Modular Backend Refactoring
- **Goal:** Extract business logic from Colab notebooks into a production-grade modular Python codebase.
- **Modules Created:**
  - `src/retrieval.py`: Clean keyword ticket retrieval module with schema and type validation.
  - `src/prompt_builder.py`: Externalized prompt builder reading template instructions from `prompts/evidence_prompt.txt`.
  - `src/gemini_client.py`: API client integrating the new `google-genai` SDK with exponential backoff retry handling for transient errors.
  - `src/memo_service.py`: Orchestration layer coordinating `retrieve_tickets` ➔ `build_evidence_prompt` ➔ `gemini_client.generate_response`.
- **Dataset & Testing:**
  - Created `data/sample/synthetic_tickets.csv`.
  - Implemented unit test suite in `tests/` covering retrieval, prompt building, API client retries, and service orchestration.

---

### 🟢 Sprint 2: Semantic Retrieval & Vector Search Engine
- **Goal:** Upgrade ticket retrieval from keyword substring matching to vector semantic search using sentence embeddings without changing the rest of the pipeline.
- **Architecture Updates:**
  - Created `src/config.py` for central hyper-parameter configuration (`MODEL_NAME`, `DEFAULT_TOP_K`, `DEFAULT_SIMILARITY_THRESHOLD`).
  - Created `src/similarity_engine.py` providing pure NumPy vector math for cosine similarity ($\mathbf{E}_{\text{tickets}} \cdot \mathbf{e}_{\text{query}}^T$) and top-k candidate ranking without FAISS dependencies.
  - Created `src/embedding_generator.py` implementing a model singleton (`SentenceTransformer('all-MiniLM-L6-v2')`) and an in-memory ticket embedding cache keyed by dataset MD5 hash.
  - Created `src/semantic_search.py` coordinator module.
  - Refactored `src/retrieval.py` to route through semantic search while maintaining 100% backwards compatibility with `memo_service.py`.
- **Pipeline Demo & Test Coverage:**
  - Implemented `tests/run_pipeline_demo.py` multi-scenario integration tool.
  - Expanded test suite to **46 unit tests** with **94%+ code coverage**, all passing deterministically.

---

### 🟢 Sprint 3: Deterministic Evidence Scoring Engine & Calibration
- **Goal:** Replace subjective model guessing with rule-based evidence scoring, validation filtering, honest recommendation prompts, and retrieval transparency.
- **Sprint 3.0 (EvidenceScoringEngine):**
  - Created `src/evidence_scoring.py` to evaluate ticket volume, sentiment consistency, severity keywords, recency, and user diversity on a 0–100 numerical scale and confidence tier (Low, Moderate, High).
- **Sprint 3.1 (Evidence Validation Layer):**
  - Created `src/evidence_validation.py` to filter candidates against `EVIDENCE_SIMILARITY_THRESHOLD` (0.60).
  - Integrated into `src/memo_service.py` to short-circuit zero-evidence proposals cleanly.
- **Sprint 3.2 (Honest Recommendation Prompt Guidelines):**
  - Updated `prompts/evidence_prompt.txt` to enforce strict honesty guidelines for weak or conflicting evidence.
- **Sprint 3.3 (Universal Data Ingestion & Schema Normalization):**
  - Created `src/preprocessing/` package providing an input-agnostic ingestion layer for **CSV, Excel (.xlsx), JSON, TXT, and PDF** files.
  - Implemented modular parsers (`csv_parser.py`, `excel_parser.py`, `json_parser.py`, `text_parser.py`, `pdf_parser.py`), file type detector (`detector.py`), schema normalizer (`normalizer.py`), custom exceptions (`exceptions.py`), and public loader API (`loader.py`).
  - Standardized all raw inputs into canonical `[ticket_id, created_at, topic, message]` DataFrames before entering downstream retrieval.
  - Created multi-format sample datasets in `data/sample/` (`synthetic_tickets.xlsx`, `synthetic_tickets.json`, `synthetic_tickets.txt`, `synthetic_tickets.pdf`).
  - Expanded test suite to **102 unit & regression tests** with **100% pass rate**.

---

### 🟢 Sprint 3.4 (Source Connectors — External Service Adapter Layer)
- **Goal:** Extend the ingestion pipeline from file-only sources to external service APIs (Google Sheets, Notion, Zendesk, Intercom) without changing any downstream pipeline code.
- **Architecture:**
  - Created `src/connectors/` package — parallel to `src/preprocessing/`, providing a unified adapter layer for external data sources.
  - `BaseConnector` (ABC): Abstract interface enforcing `fetch() → raw DataFrame` contract. Provides inherited `_retry()` with exponential `_backoff()` for transient API failure resilience.
  - `ConnectorConfig` (dataclass): Shared configuration (timeout, retries, backoff_factor, batch_size) with sensible defaults.
  - `auth.py`: Centralized `resolve_credential()` helper — explicit value → `.env` variable → `AuthenticationError` with clear instructions.
  - `pagination.py`: Shared pagination utilities — `paginate_offset()` (Zendesk-style) and `paginate_cursor()` (Notion/Intercom-style).
  - `exceptions.py`: Connector exception hierarchy (`ConnectorError`, `AuthenticationError`, `RateLimitError`, `EmptyResponseError`, `PaginationError`).
- **Connectors Implemented:**
  - `GoogleSheetsConnector`: Google Sheets API v4 via `gspread` + `google-auth` service account authentication.
  - `NotionConnector`: Notion API v1 via `requests` with full property type extraction (title, rich_text, select, multi_select, date, number, checkbox, etc.).
  - `ZendeskConnector`: Zendesk REST API v2 via `requests` with email/token basic auth and `next_page` URL pagination.
  - `IntercomConnector`: Intercom REST API v2.11 via `requests` with bearer token auth and cursor-based pagination.
- **Orchestration:**
  - `loader.py` with `load_connector()` entry point — mirrors `preprocessing/loader.py` exactly: registry lookup → `connector.fetch()` → `normalize_data()` → standard DataFrame.
  - Connectors return **raw data only**; normalization is the loader's responsibility (zero coupling between connectors and preprocessing).
- **API Symmetry:**
  - Files: `load_data("tickets.csv")` → normalized DataFrame.
  - APIs: `load_connector("google_sheets", spreadsheet_id="...")` → normalized DataFrame.
- **Testing:** Expanded test suite to **152 unit & regression tests** with **100% pass rate**. All connector tests fully mocked — no real credentials required.

---

### 🟢 Sprint 3.6: Advanced Evidence Intelligence
- **Goal:** Provide deterministic deduplication, semantic clustering, historical calibration, and multi-criteria product decision logic.
- **Modules Created:**
  - `src/evidence_deduplication.py`: Exact normalization and dense vector semantic deduplication with representative selection strategies (`longest`, `highest_similarity`, `earliest`) and comprehensive telemetry.
  - `src/evidence_clustering.py`: Agglomerative semantic clustering with medoid theme identification.
  - `src/historical_calibration.py`: Storage-abstracted historical outcome tracker with Brier score and Expected Calibration Error (ECE) metrics.
  - `src/decision_engine.py`: Deterministic multi-criteria decision engine evaluating Evidence Score, Effort, Impact, Strategic Alignment, Cost, and Risk.
  - `src/memo_service.py` & `src/prompt_builder.py`: Extended pipeline orchestration producing structured Decision Memos.
- **Testing:** Expanded test suite to **188 automated unit & integration tests** with 100% pass rate.

---

### 🟢 Sprint 4: Real-World Dataset Validation & Benchmarking
- **Goal:** Rigorously validate the platform on real-world customer feedback datasets, perform multi-scale benchmarking, quality evaluations, and threshold tuning.
- **Architecture & Modules Created:**
  - `data/evaluation/`: Ingested real-world public datasets across Google Play Reviews, GitHub Issues, Customer Support Tweets, and Amazon Product Reviews. Config-driven registry via `data/evaluation/datasets.json`.
  - `src/evaluation/adapters.py`: Decoupled `BaseDatasetAdapter` subclasses transforming domain schemas to canonical schema without modifying existing pipeline code.
  - `src/evaluation/quality.py`: Evaluator for Retrieval (Precision@5: 0.467, Recall@5: 0.944, MRR: 1.000, MAP: 1.000), Deduplication (F1: 0.571), and Calibration (Brier: 0.000 / 0.042, ECE: 0.000 / 0.065).
  - `src/evaluation/benchmark.py`: Multi-scale benchmarking engine (1K, 5K, 10K, 25K, 50K records). Demonstrated NumPy retrieval at **40.0 ms** for 50,000 records, proving that FAISS is not needed for sub-100K datasets.
  - `src/evaluation/threshold_optimizer.py`: Parameter sensitivity sweeps confirming optimal thresholds (`EVIDENCE_SIMILARITY_THRESHOLD = 0.60`, `DEDUPLICATION_SIMILARITY_THRESHOLD = 0.85`, `CLUSTERING_DISTANCE_THRESHOLD = 0.35`).
  - `src/evaluation/e2e_validator.py` & `evaluation/examples/`: Generated verified Markdown decision memo artifacts across all domains.
  - `tests/test_evaluation_edge_cases.py`: Hardened error handling for multilingual text, emojis, missing timestamps, missing topics, and extreme string lengths.
  - `docs/sprint_4_evaluation_report.md`: Comprehensive evaluation report and benchmark tables.
- **Testing:** Expanded test suite to **212+ automated tests** with 100% pass rate and 0 regressions.


