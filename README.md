

---

## 🧩 Customization & Extension

### Swap LLMs
- Install your preferred LLM package (e.g., `pip install langchain-openai`)
- Edit `app/models/model_loader.py` to use your LLM

### Change Embeddings
- Edit `app/models/embeddings.py` to use a different embedding model
- Update the document processor if needed

### Add New Tools
- Add a new function in `app/core/agent.py`
- Register it in the agent's tool routing logic

---

## � Configuration

<!--
  RAG-Agentic-HybridSearch
  A compact RAG reference app combining FAISS, BM25, RRF fusion, and a deterministic tool executor.
-->

# RAG-Agentic-HybridSearch

RAG-Agentic-HybridSearch is a small, modular Retrieval-Augmented Generation (RAG)
reference implementation. It provides a Streamlit chat interface to ask questions over
uploaded PDF documents and (optionally) live web search results. The project focuses on
deterministic, markdown-friendly answers, observable terminal logs, and an easy-to-extend
code structure.

Highlights:
- In-memory FAISS vector store + BM25 sparse retriever
- Reciprocal Rank Fusion (RRF) + optional reranking for high-quality candidates
- Lightweight executor that returns direct Markdown answers (no complex ReAct loops)
- Streamlit UI with processing progress, chat history, and citation display

---

## Table of contents

- Features
- Architecture overview
- Quickstart (Windows)
- Common workflows
- Configuration & tuning
- Development notes
- Troubleshooting
- Security & privacy
- Roadmap
- Contributing
- License

---

## Features

- Upload and chunk PDF documents with metadata
- Build FAISS (dense) and BM25 (sparse) indices
- Hybrid retrieval using RRF fusion and cross-encoder reranking
- Context compression and LLM answer generation
- Simple tool executor that returns the tool output as the final Markdown answer
- Streamlit UI with logging that prints to the terminal when Streamlit is started from a shell

---

## Architecture overview

- `app/core/document_processor.py` — PDF ingestion, text extraction, chunking, and index builder
- `app/core/rag_pipeline.py` — dense/sparse retrieval, fusion, reranking, context compression, answer generation
- `app/core/agent.py` — simplified tool executor and tool functions (document QA, summarizer, web search)
- `app/models` — LLM and embedding loaders
- `app/ui/components.py` — Streamlit UI components and chat rendering
- `streamlit_app.py` — entrypoint; configures logging and session state

The code favors small, single-responsibility modules so you can swap components (LLM,
embeddings, retriever) independently.

---

## Quickstart (Windows)

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add required API keys (example):

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

4. Start the app (run from a terminal to see logs):

```powershell
streamlit run streamlit_app.py
```

5. Use the sidebar to upload PDFs, adjust chunk size/overlap, click "Process Documents",
   then ask questions in the chat box.

Notes:
- Restart the app after changing embeddings or LLM implementation.
- Start Streamlit from a shell to ensure logs are printed to stdout.

---

## Common workflows

- Summarize uploaded documents: "Summarize the uploaded documents."
- Extract contractual fields: "What is the notice period for termination?"
- Investigate a topic across documents: "Show passages that mention `data retention` and summarize."
- Combine web + docs: use the web search tool (requires Tavily key) to include fresh web context.

---

## Configuration & tuning

- Chunk size and overlap: set in the sidebar when ingesting documents. Typical defaults: 400–800 tokens per chunk, 50–100 overlap.
- Retriever `k` and reranker thresholds: configurable in `app/core/rag_pipeline.py`.
- Debug mode: toggleable in the UI to surface timing and intermediate info.

If you change embedding or LLM models, re-run document processing to rebuild FAISS and BM25 indices.

---

## Development notes

Swap components:
- LLM: edit `app/models/model_loader.py` to return a different LangChain-compatible LLM instance.
- Embeddings: edit `app/models/embeddings.py`, then reprocess documents.
- Add tools: add functions in `app/core/agent.py` and add routing in the executor.

Linting / basic checks:

```powershell
python -m py_compile app\core\agent.py app\core\rag_pipeline.py app\ui\components.py
pip install black isort
black --check .
isort --check-only .
```

Recommended CI steps:
- `python -m py_compile` for changed Python files
- `black --check` and `isort --check-only`
- small smoke test that runs the app and submits a minimal query against a sample doc

---

## Troubleshooting

- No answer returned:
  - Make sure you processed documents (check sidebar and `processed_files`).
  - Confirm API keys are set and valid.
  - Inspect terminal logs for exceptions; start Streamlit from a terminal to see logs.

- Logs not visible:
  - Start Streamlit from a shell/terminal; the app configures logging to write to stdout for visibility.

- Index build fails or returns empty results:
  - Ensure PDFs contain selectable text (OCR is required for scanned images).
  - Check chunk size and filtering: very small chunks may be discarded.

---

## Security & privacy

- Uploaded documents are stored in memory and session state by default — they are not persisted to disk by the app.
- Do not commit API keys to source control. Use environment variables or a secrets manager in production.

---

## Roadmap / ideas

- Persisted vector stores (Milvus / Weaviate / disk-backed FAISS) for larger datasets
- Unit tests for document processing and retrieval logic
- Role-based access control and quotas for multi-tenant deployment

---

## Contributing

1. Fork and create a branch: `git checkout -b feature/your-feature`
2. Add tests for new logic where applicable
3. Run formatters: `black`, `isort`
4. Open a PR describing the change and rationale

Follow PEP8 and keep changes small and reviewable.


