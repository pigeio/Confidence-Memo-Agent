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
