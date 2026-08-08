# Production Architecture Documentation

This document describes the module architecture of the **Confidence Memo Agent**, detailing the purpose, inputs, outputs, responsibilities, and dependencies of each component in the system.

---

## 📐 High-Level Architecture Diagram

```text
                        ┌────────────────────────┐
                        │ User Feature Proposal  │
                        └───────────┬────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        Retrieval Layer (src/)                          │
│                                                                        │
│   retrieval.py (Public Wrapper)                                        │
│        │                                                               │
│        ▼                                                               │
│   semantic_search.py (Search Coordinator)                              │
│        ├──> embedding_generator.py (Singleton Model & Dataset Cache)   │
│        └──> similarity_engine.py   (Pure NumPy Cosine Vector Search)   │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ Matched Tickets DataFrame
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Prompt Builder Layer (src/)                          │
│                                                                        │
│   prompt_builder.py  <──> prompts/evidence_prompt.txt                  │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ Formatted Prompt String
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      Gemini Client Layer (src/)                        │
│                                                                        │
│   gemini_client.py   <──> Google Gemini API (gemini-3.5-flash)          │
└───────────────────────────┬────────────────────────────────────────────┘
                            │ API Text Response
                            ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer (src/)                         │
│                                                                        │
│   memo_service.py    ───> Returns Evidence Memo String               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Module Specifications

### 1. `config.py`
- **Purpose:** Centralized configuration management and model default parameters.
- **Inputs:** Environment variables (`EMBEDDING_MODEL_NAME`, `DEFAULT_TOP_K`, `DEFAULT_SIMILARITY_THRESHOLD`).
- **Outputs:** Module-level constants (`MODEL_NAME`, `DEFAULT_TOP_K`, `DEFAULT_SIMILARITY_THRESHOLD`).
- **Responsibilities:**
  - Read environment overrides for embedding model selection and search thresholds.
  - Provide fallback default constants across the codebase.
- **Dependencies:** `os`.

---

### 2. `embedding_generator.py`
- **Purpose:** Vector embedding generation, neural model lifecycle management, and ticket dataset caching.
- **Inputs:** Raw text strings or pandas DataFrame (`topic`, `message` columns).
- **Outputs:** 2D NumPy array of L2-normalized dense embeddings (`np.ndarray` of shape `(N, 384)`).
- **Responsibilities:**
  - Maintain a singleton instance of `SentenceTransformer('all-MiniLM-L6-v2')` to avoid re-loading weights on every query.
  - Calculate MD5 dataset fingerprints for support ticket DataFrames.
  - Maintain an in-memory embedding cache dictionary to reuse ticket vector matrices instantly across queries.
- **Dependencies:** `sentence_transformers`, `pandas`, `numpy`, `hashlib`, `src.config`.

---

### 3. `similarity_engine.py`
- **Purpose:** Pure NumPy vector math for computing cosine similarity and candidate ranking.
- **Inputs:** Ticket embedding matrix (`N, D`), query vector (`1, D`), `top_k` integer, `threshold` float.
- **Outputs:** List of `(index, similarity_score)` tuples sorted in descending order of relevance.
- **Responsibilities:**
  - Compute matrix dot-product similarity ($\mathbf{E}_{\text{tickets}} \cdot \mathbf{e}_{\text{query}}^T$).
  - Rank candidate indices descending by similarity score.
  - Filter out candidates scoring below the similarity threshold up to `top_k`.
  - Validate array shapes and dimensions without loading heavy neural network dependencies.
- **Dependencies:** `numpy`, `src.config`.

---

### 4. `semantic_search.py`
- **Purpose:** Search coordinator combining `EmbeddingGenerator` and `SimilarityEngine`.
- **Inputs:** Support tickets DataFrame, query string or keyword list, optional `top_k` and `threshold` parameters.
- **Outputs:** Filtered subset `pd.DataFrame` containing the most relevant tickets sorted by semantic similarity.
- **Responsibilities:**
  - Normalize query input types (handling string queries or keyword lists).
  - Invoke `EmbeddingGenerator` to encode the dataset and query string.
  - Invoke `SimilarityEngine` to compute vector similarity and pick top-k indices.
  - Return a sliced DataFrame of relevant ticket rows.
- **Dependencies:** `pandas`, `src.embedding_generator`, `src.similarity_engine`, `src.config`.

---

### 5. `retrieval.py`
- **Purpose:** Public backward-compatible interface wrapper for ticket retrieval.
- **Inputs:** `df` (`pd.DataFrame`), `keywords` (`list[str] | str`).
- **Outputs:** `relevant_tickets` (`pd.DataFrame`).
- **Responsibilities:**
  - Expose a simple, clean `retrieve_tickets` entry point.
  - Delegate execution directly to `semantic_search.search_tickets`.
- **Dependencies:** `pandas`, `src.semantic_search`.

---

### 6. `prompt_builder.py`
- **Purpose:** Formats retrieved tickets and constructs the evidence analysis prompt.
- **Inputs:** `proposal` (`str`), `df_tickets` (`pd.DataFrame`), optional `template_path` (`str`).
- **Outputs:** Formatted prompt string (`str`).
- **Responsibilities:**
  - Format ticket DataFrame rows into clear, structured ticket strings.
  - Read prompt template from file (`prompts/evidence_prompt.txt`) or default inline template.
  - Substitute `{proposal}` and `{tickets_text}` placeholders.
- **Dependencies:** `os`, `pandas`.

---

### 7. `gemini_client.py`
- **Purpose:** Interface client for Google Gemini LLM API interactions.
- **Inputs:** Prompt string (`str`).
- **Outputs:** Generated Markdown text response (`str`).
- **Responsibilities:**
  - Manage `google.genai.Client` lifecycle using `GEMINI_API_KEY`.
  - Execute API calls against `gemini-3.5-flash`.
  - Implement exponential backoff retry logic for transient errors (`429`, `500`, `503`, `504`).
  - Fail fast on terminal errors (`400`, `403`).
- **Dependencies:** `google.genai`, `google.genai.errors`, `os`, `time`, `logging`.

---

### 8. `memo_service.py`
- **Purpose:** End-to-end orchestration layer for the Confidence Memo Agent.
- **Inputs:** `df` (`pd.DataFrame`), `keywords` (`list[str]`), `proposal` (`str`), optional `template_path` (`str`).
- **Outputs:** Raw Evidence Memo Markdown text (`str`).
- **Responsibilities:**
  - Execute pipeline sequence: `retrieve_tickets` ➔ `build_evidence_prompt` ➔ `gemini_client.generate_response`.
  - Provide a single unified method call for application consumption.
- **Dependencies:** `pandas`, `src.retrieval`, `src.prompt_builder`, `src.gemini_client`.
