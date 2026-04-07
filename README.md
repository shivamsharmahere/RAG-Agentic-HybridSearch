# Advanced RAG Agent

A production-ready Retrieval-Augmented Generation (RAG) based AI Chatbot that answers user queries using PDF documents as knowledge sources. Features hybrid search, cross-encoder reranking, and an intelligent agent with tool execution capabilities.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features Explained](#features-explained)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **PDF RAG** | Upload and process PDF documents with metadata extraction |
| **Hybrid Search** | Combine dense (FAISS) and sparse (BM25) retrieval for comprehensive results |
| **ReRanked Results** | Cross-encoder reranking for improved answer quality |
| **Context Compression** | Optimize context window by compressing relevant passages |
| **ReAct Agent** | Intelligent tool executor with document QA, summarization, and web search capabilities |
| **Conversational Memory** | Maintain chat history for contextual follow-up questions |
| **Metadata Citations** | Source attribution with file name and page references |
| **Streamlit UI** | Intuitive web interface for document management and chat |

---

## Architecture Overview

```
RAG1/
├── app/
│   ├── core/
│   │   ├── document_processor.py   # PDF ingestion, text extraction, chunking
│   │   ├── rag_pipeline.py          # Hybrid retrieval, fusion, reranking, answer generation
│   │   └── agent.py                 # Tool executor and agent logic
│   ├── models/
│   │   ├── model_loader.py          # LLM loader (Groq, OpenAI, Anthropic)
│   │   └── embeddings.py           # Sentence transformer embeddings
│   ├── ui/
│   │   └── components.py            # Streamlit UI components
│   ├── config/
│   │   └── constants.py            # Application constants
│   └── utils/
│       └── session_state.py         # Session state management
├── streamlit_app.py                  # Application entry point
└── requirements.txt                 # Dependencies
```

### Core Components

| Component | Purpose |
|-----------|---------|
| `document_processor.py` | PDF parsing, text splitting, metadata extraction, and index building |
| `rag_pipeline.py` | Dense/sparse retrieval, RRF fusion, cross-encoder reranking, context compression |
| `agent.py` | Simplified tool executor with document QA, summarizer, and web search tools |
| `model_loader.py` | LLM initialization supporting multiple providers (Groq, OpenAI, Anthropic) |
| `embeddings.py` | Sentence transformer embeddings for semantic search |

---

## Prerequisites

| Requirement | Description |
|-------------|-------------|
| **Python** | 3.9 or higher |
| **API Keys** | Groq API key (required), Tavily API key (optional, for web search) |
| **Operating System** | Windows, macOS, or Linux |
| **Memory** | 4GB+ RAM recommended for embeddings and LLM inference |

### Required API Keys

- **Groq API Key** — Required for LLM inference. Get one at [groq.com](https://groq.com)
- **Tavily API Key** — Optional, for web search functionality. Get one at [tavily.com](https://tavily.com)

---

## Installation

### Windows (PowerShell)

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/rag1.git
   cd rag1
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

5. **Start the application**
   ```powershell
   streamlit run streamlit_app.py
   ```

The application will open in your default browser at `http://localhost:8501`.

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes | — | API key for Groq LLM inference |
| `TAVILY_API_KEY` | No | — | API key for web search functionality |
| `OPENAI_API_KEY` | No | — | Alternative LLM provider |
| `ANTHROPIC_API_KEY` | No | — | Alternative LLM provider |
| `MAX_CONTEXT_TOKENS` | No | 1800 | Maximum tokens for context window |
| `RERANK_KEEP` | No | 8 | Number of results to keep after reranking |
| `DEFAULT_CHUNK_SIZE` | No | 400 | Default chunk size in tokens |
| `DEFAULT_CHUNK_OVERLAP` | No | 60 | Chunk overlap in tokens |
| `DEBUG` | No | False | Enable debug logging |

### Runtime Configuration

Configure via the Streamlit sidebar:

- **Chunk Size** — Tokens per chunk (recommended: 400–800)
- **Chunk Overlap** — Overlap between chunks (recommended: 50–100)
- **Top-K Retriever** — Number of documents to retrieve
- **LLM Model** — Select from available Groq models (Llama 3.3, etc.)

---

## Usage

### Getting Started

1. **Launch the app**: `streamlit run streamlit_app.py`
2. **Upload PDFs**: Use the sidebar to upload one or more PDF documents
3. **Process documents**: Click "Process Documents" to build indices
4. **Ask questions**: Type your question in the chat input

### Common Use Cases

| Task | Example Query |
|------|---------------|
| Summarize documents | "Summarize the uploaded documents." |
| Extract specific information | "What is the notice period for termination?" |
| Research across documents | "Show passages about data retention and summarize." |
| Web +文档 | "Search the web for latest developments on X" |

### Demo Workflows

#### Level 1 — Basic PDF RAG
1. Start Streamlit: `streamlit run streamlit_app.py`
2. Upload PDFs in the sidebar
3. Click "Process Documents"
4. Ask questions in the chat box

#### Level 2 — Production Pipeline
- Monitor logs in terminal to see index-building and pipeline steps
- Use debug mode for timing and intermediate information

#### Level 3 — Conversational Memory
- Ask a follow-up question referencing prior context
- Chat history is preserved in session state

#### Level 4 — Metadata Filtering
- Upload multiple PDFs with different names
- View source citations in the expander (file and page info)

#### Level 5 — Agent Tools
- Use tool-based queries: "Summarize documents", "Search the web for X"

---

## Features Explained

### PDF Processing Pipeline

1. **Text Extraction** — Uses PyMuPDF to extract text from PDF pages
2. **Chunking** — Splits text into overlapping chunks with configurable size
3. **Metadata** — Captures filename, page number, chunk index for citations

### Hybrid Retrieval

1. **Dense Retrieval** — FAISS vector store with sentence transformer embeddings
2. **Sparse Retrieval** — BM25 ranking for keyword-based matching
3. **RRF Fusion** — Reciprocal Rank Fusion combines results from both retrievers
4. **Cross-Encoder Reranking** — Reorders results using a cross-encoder model

### Context Compression

- Uses LangChain's ContextualCompressionExtractor
- Wraps base retriever with LLM-based compression
- Extracts relevant content based on the query

### Agent System

- **SimpleToolExecutor** — Routes queries to appropriate tools
- **Available Tools**:
  - Document QA — Answers questions using uploaded documents
  - Summarizer — Generates document summaries
  - Web Search — Uses Tavily for fresh web information

---

## API Reference

### Core Modules

#### `app/core/document_processor.py`

```python
class DocumentProcessor:
    def load_pdf(self, file_path: str) -> List[Document]
    def chunk_documents(self, documents: List[Document], chunk_size: int, overlap: int) -> List[Document]
    def build_faiss_index(self, documents: List[Document]) -> FAISS
    def build_bm25_index(self, documents: List[Document]) -> BM25Retriever
```

#### `app/core/rag_pipeline.py`

```python
class RAGPipeline:
    def __init__(self, retriever, llm, compressor)
    def get_relevant_documents(self, query: str) -> List[Document]
    def generate_answer(self, query: str, context: List[Document]) -> str
    def hybrid_retrieve(self, query: str) -> List[Document]
```

#### `app/core/agent.py`

```python
class SimpleToolExecutor:
    def execute(self, query: str, tools: List[BaseTool], memory: Any) -> str
    def route_to_tool(self, query: str, available_tools: List[str]) -> str
```

### Models

#### `app/models/model_loader.py`

```python
def load_llm(provider: str = "groq", model_name: str = "llama-3.3-70b-versatile") -> BaseLLM
def get_available_models() -> List[str]
```

#### `app/models/embeddings.py`

```python
def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings
```

---

## Development

### Coding Standards

- Follow PEP 8 style guide
- Use Black for formatting: `black .`
- Use isort for imports: `isort .`
- Type hints required for all functions

### Running Tests

```powershell
pytest tests/
```

### Linting

```powershell
pip install black isort mypy
black --check .
isort --check-only .
mypy .
```

### Recommended CI Pipeline

1. `python -m py_compile` — Syntax validation
2. `black --check` — Format validation
3. `isort --check-only` — Import sorting
4. `pytest` — Run test suite
5. Smoke test — Verify app starts and processes a sample query

### Adding New Features

- **Swap LLM**: Edit `app/models/model_loader.py`
- **Swap Embeddings**: Edit `app/models/embeddings.py`, then reprocess documents
- **Add Tools**: Add functions in `app/core/agent.py` and update routing logic

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No answer returned | Ensure documents are processed; verify API keys are set |
| Empty search results | Check that PDFs contain selectable text (OCR required for scanned images) |
| Index build fails | Verify chunk size is not too small; PDFs must have extractable text |
| API errors | Confirm API keys are valid and not expired |
| Logs not visible | Run Streamlit from a terminal (not IDE) to see stdout logs |

### Debug Mode

Enable debug mode in the sidebar to see:
- Timing information for each pipeline stage
- Intermediate retrieval results
- Reranking scores
- Context compression details

### Getting Help

1. Check terminal logs for exception details
2. Enable debug mode for verbose output
3. Verify API keys are correctly set in `.env`
4. Ensure all dependencies are installed

---

## Roadmap

### Planned Features

- [ ] **Persistent Vector Stores** — Milvus, Weaviate, or disk-backed FAISS for larger datasets
- [ ] **Unit Tests** — Comprehensive test coverage for document processing and retrieval
- [ ] **Role-Based Access Control** — Multi-tenant deployment with quotas
- [ ] **Advanced Memory** — Long-term memory with retrieval-augmented memory chains
- [ ] **LangChain Agent Integration** — Full ReAct agent with dynamic tool selection

### Improvements

- [ ] Docker containerization
- [ ] Cloud deployment configurations
- [ ] Webhook integrations for external triggers
- [ ] Custom document loaders (DOCX, CSV, etc.)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Add tests for new functionality
4. Run formatters: `black .` and `isort .`
5. Commit your changes with clear commit messages
6. Push to your branch: `git push origin feature/your-feature`
7. Open a Pull Request

For major changes, please open an issue first to discuss the proposed approach.

---

*Built with Streamlit, LangChain, FAISS, and Groq.*
