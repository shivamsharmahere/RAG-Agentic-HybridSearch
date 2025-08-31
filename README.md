# Advanced RAG Agent 🧠

A production-worthy Retrieval-Augmented Generation (RAG) based AI Chatbot that answers user queries using PDF documents as knowledge sources.

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

## Assignment: Five Levels (mapping to this repo)

Below is a concise mapping of the assignment's five levels to the current codebase. Each level shows status, the files to inspect, quick demo steps, and short notes about limitations or next steps to reach a full "production" implementation.

### Level 1 — PDF RAG with Semantic Search
- Status: Done
- What is implemented: PDF parsing, text splitting, embeddings, FAISS vector store, top-k retrieval, and LLM answer generation.
- Key files: `app/core/document_processor.py`, `app/models/embeddings.py`, `app/models/model_loader.py`, `app/core/rag_pipeline.py`
- How to demo:
  1. Start Streamlit: `streamlit run streamlit_app.py`
  2. Upload one or more PDFs in the sidebar and click "Process Documents"
  3. Ask a question in the chat box; the RAG pipeline will retrieve and answer.
- Notes: Works out-of-the-box after installing dependencies and setting GROQ API key.

### Level 2 — Production-Ready RAG with LangChain (modular pipeline)
- Status: Done (modularized)
- What is implemented: LangChain components are used (document loaders, text splitter, retrievers, and prompt templates). The project is organized into modular components (document processing, retrieval pipeline, agent/tools, UI).
- Key files: `app/core/document_processor.py`, `app/core/rag_pipeline.py`, `app/core/agent.py`, `app/ui/components.py`, `app/main.py`
- How to demo: same as Level 1; monitor logs in the terminal to see index-building and pipeline steps.
- Notes: Pipeline is modular but still tightly integrated with Streamlit session state; decoupling logic from UI will improve testability.

### Level 3 — Conversational Memory Support
- Status: Partial / Done (basic support)
- What is implemented: Chat history and a memory object are present; user and assistant messages are appended to `st.session_state.chat_history`, and the executor writes to `memory.chat_memory` when invoking tools.
- Key files: `app/core/agent.py`, `app/ui/components.py`, `app/main.py`, `app/utils/session_state.py` (initialization)
- How to demo:
  1. Process documents.
  2. Ask a question, then ask a follow-up that refers to prior context; the UI shows chat history and the executor keeps a memory buffer.
- Notes: Memory is present but relatively simple (ConversationBuffer). For advanced conversational behavior (long-term memory, retrieval-augmented memory), consider integrating a persistent conversation store or more structured memory chains.

### Level 4 — Metadata Tagging and Filtering
- Status: Done
- What is implemented: Document metadata (file name, page, chunk info) is attached at ingestion; compression returns citation metadata and the UI displays citations with file and page information.
- Key files: `app/core/document_processor.py`, `app/core/rag_pipeline.py`, `app/ui/components.py`
- How to demo:
  1. Upload multiple PDFs with different names.
  2. Ask a question that should be answered by a specific document; inspect the Sources/expander to see file/page citations.
- Notes: Metadata extraction is available. If you need advanced filters (restrict to one document by name or date ranges), add metadata-based filtering hooks into the retrievers before fusion.

### Level 5 — Agent-based Chatbot with Tool Use
- Status: Partial
- What is implemented: A simplified tool executor (`SimpleToolExecutor`) routes queries to document QA, summarizer, or web search tools. It provides deterministic Markdown outputs and logging. This intentionally bypasses the full LangChain ReAct agent to avoid ReAct parsing complexity.
- Key files: `app/core/agent.py`, `app/core/rag_pipeline.py`, `app/ui/components.py`
- How to demo:
  1. Upload documents and process.
  2. Ask high-level queries like "Summarize the documents" or "Search the web for latest X" to see tool routing.
- Notes & gaps: The current executor is simpler than a full LangChain AgentExecutor. To meet the assignment's Level 5 expectations strictly (dynamic tool selection, multi-step chaining using LangChain agents), either:
  - Reintroduce LangChain agents (AgentExecutor + tools) with structured tool outputs and a safe prompt template, or
  - Enhance `SimpleToolExecutor` to support multi-step plans and an action-observation loop with robust parsing and fallbacks.

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

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact

- Maintainer: RAG Developer <example@example.com>
- Issues: https://github.com/shivamsharmahere/RAG-Agentic-HybridSearch/issues
