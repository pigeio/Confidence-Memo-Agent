# Product Roadmap — Confidence Memo Agent

This document tracks completed phases and future engineering sprints for the Confidence Memo Agent.

---

## 🟢 Completed Sprints

### ✅ Sprint 0: Proof-of-Concept Prototype
- Google Colab prototype demonstrating evidence-based confidence calibration.
- Synthetic dataset of 20 customer tickets.

### ✅ Sprint 1: Backend Modularization
- Decoupled modular architecture (`retrieval`, `prompt_builder`, `gemini_client`, `memo_service`).
- Comprehensive unit test suite.
- Integration with the new `google-genai` SDK and retry handling.

### ✅ Sprint 2: Semantic Retrieval Engine
- Vector embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`.
- Model singleton instantiation & dataset MD5 embedding cache.
- Pure NumPy Cosine Similarity & top-k ranking engine.
- End-to-end multi-scenario pipeline demonstration script (`tests/run_pipeline_demo.py`).
- 46 unit tests with 94%+ code coverage.

---

## 🔮 Future Sprints

### ⬜ Sprint 3: Evidence Scoring Engine
- Quantitative Evidence Score (0–100 scale).
- Analytical breakdown based on ticket volume, sentiment consistency, urgency, recency, and user diversity.

### ⬜ Sprint 4: Decision Memo Engine
- Secondary deep-dive analysis module for prioritised features.
- Evaluates technical effort, compute impact, memory/battery footprint, engineering risks, pros/cons, and final trade-off recommendations.

### ⬜ Sprint 5: Historical Calibration & Accuracy
- Historical tracking of feature outcomes vs. past AI confidence recommendations to calibrate scoring accuracy over time.

### ⬜ Sprint 6: Streamlit Web UI
- Interactive web app for product managers to query features, view visual Evidence Memos, inspect ticket quotes, and request Decision Memos.

### ⬜ Sprint 7: Multi-Source Integrations
- Connectors for Jira, Slack, Linear, Zendesk, NPS surveys, and session replays to enrich the evidence base.
