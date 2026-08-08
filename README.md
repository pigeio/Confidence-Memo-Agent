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
               User Feature Proposal
                         │
                         ▼
                 Semantic Retrieval
             (Sentence Transformers & NumPy)
                         │
                         ▼
                   Prompt Builder
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
                  Evidence Memo
            (Calibrated Product Report)
```

---

## ✨ Features Completed

### ✅ Sprint 1: Backend Modularization
- **Modular Retrieval Interface (`src/retrieval.py`):** Clean separation for ticket querying with schema validation.
- **Prompt Builder (`src/prompt_builder.py`):** Externalized prompt template engine (`prompts/evidence_prompt.txt`) with fallback support.
- **Gemini API Client (`src/gemini_client.py`):** Integration with the new `google-genai` SDK featuring exponential backoff retries for transient errors.
- **Memo Service (`src/memo_service.py`):** Unified pipeline orchestration service.

### ✅ Sprint 2: Semantic Retrieval Engine
- **Dense Vector Embeddings (`src/embedding_generator.py`):** Uses `SentenceTransformer('all-MiniLM-L6-v2')` with a model singleton to avoid re-initialization overhead.
- **DataFrame Embedding Cache:** In-memory dataset hash caching (MD5) to avoid re-encoding ticket DataFrames across queries.
- **NumPy Cosine Similarity (`src/similarity_engine.py`):** Fast, zero-FAISS vector dot product search over pre-normalized embeddings.
- **Top-k Relevance Ranking:** Ranks and filters candidates based on configurable similarity thresholds.
- **End-to-End Pipeline Demo (`tests/run_pipeline_demo.py`):** Automated multi-scenario integration script.

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
