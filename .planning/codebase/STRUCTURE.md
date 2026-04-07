# Codebase Structure

**Analysis Date:** 2026-04-07

## Directory Layout

```
[project-root]/
├── .planning/          # GSD planning documents (auto-generated)
├── app/                # Main application source code
│   ├── config/         # Configuration constants and settings
│   ├── core/           # Business logic and processing pipeline
│   ├── models/         # ML model loading and embedding implementations
│   ├── ui/             # Streamlit UI components
│   ├── utils/          # Utility functions and helpers
│   ├── main.py         # Application entry point
│   └── __init__.py     # Package initializer
├── .env                # Environment variables (API keys) - NOT COMMITTED
├── .gitignore          # Git ignore rules
├── pyproject.toml      # Python project configuration and dependencies
├── requirements.txt    # Python package dependencies
├── README.md           # Project documentation
└── streamlit_app.py    # Legacy/alternative entry point
```

## Directory Purposes

**[app/config]:**
- Purpose: Centralized configuration management
- Contains: Constant values for application behavior
- Key files: `constants.py` - All configurable parameters

**[app/core]:**
- Purpose: Core business logic and processing pipeline
- Contains: Document processing, RAG pipeline, agent implementation
- Key files:
  - `document_processor.py` - PDF loading, chunking, and indexing
  - `rag_pipeline.py` - Advanced retrieval-augmented generation logic
  - `agent.py` - ReAct agent with tools for QA, summarization, web search

**[app/models]:**
- Purpose: Machine learning model loading and wrapper implementations
- Contains: LLM initialization, embedding models, reranker models
- Key files:
  - `model_loader.py` - Groq LLM and CrossEncoder reranker loading
  - `embeddings.py` - Qwen embedding model wrapper for LangChain

**[app/ui]:**
- Purpose: User interface components and layout
- Contains: Streamlit-based UI elements
- Key files: `components.py` - Sidebar, chat display, input handling

**[app/utils]:**
- Purpose: Cross-cutting utility functions
- Contains: Session state management and helper functions
- Key files: `session_state.py` - Streamlit session state initialization

## Key File Locations

**Entry Points:**
- `app/main.py`: Primary application entry point for Streamlit
- `streamlit_app.py`: Legacy/simple entry point (appears to be minimal)

**Configuration:**
- `app/config/constants.py`: All application constants and settings
- `.env`: Environment variables for API keys (Groq, Tavily) - gitignored

**Core Logic:**
- `app/core/document_processor.py`: Document ingestion and preparation
- `app/core/rag_pipeline.py`: Advanced RAG with query expansion, fusion, reranking
- `app/core/agent.py`: Agent creation and tool definitions (RAG, summarization, web search)

**Model Management:**
- `app/models/model_loader.py`: LLM and reranker model loading with caching
- `app/models/embeddings.py`: Custom embedding implementation

**UI Components:**
- `app/ui/components.py`: All Streamlit UI rendering and interaction logic

**Utilities:**
- `app/utils/session_state.py`: Session state initialization and management

## Naming Conventions

**Files:**
- Snake_case: `document_processor.py`, `rag_pipeline.py`, `model_loader.py`
- Descriptive names indicating purpose and responsibility

**Directories:**
- Snake_case: `app/core`, `app/models`, `app/ui`, `app/utils`
- Functional grouping by concern

**Classes:**
- PascalCase: `QwenEmbeddings` (custom embedding class)
- Descriptive class names

**Functions/Methods:**
- Snake_case: `load_and_chunk_pdfs`, `build_indexes`, `create_agent_and_tools`
- Verb-noun pattern indicating action and target

**Constants:**
- UPPER_CASE with underscores: `MAX_CONTEXT_TOKENS`, `DEFAULT_CHUNK_SIZE`
- Located in `app/config/constants.py`

## Where to Add New Code

**New Feature (e.g., additional tool):**
- Primary code: `app/core/agent.py` (add new Tool and integrate with agent)
- Tests: Not currently present; would need to establish testing pattern

**New Component/Module:**
- Implementation: `app/ui/components.py` (for UI elements) or new file in `app/core/` (for logic)
- Follow existing patterns for imports and session state usage

**Utilities:**
- Shared helpers: `app/utils/` directory
- Model-related: `app/models/` directory
- Configuration: `app/config/constants.py`

**Document Processing Enhancements:**
- `app/core/document_processor.py` (for loading/chunking changes)
- `app/core/rag_pipeline.py` (for retrieval/pipeline algorithm changes)

**New Model Integrations:**
- `app/models/` directory with new loader/wrapper classes
- Update `app/core/agent.py` to use new models if needed

## Special Directories

**[__pycache__]:**
- Purpose: Python bytecode cache directories
- Generated: Yes (automatically by Python interpreter)
- Committed: No (should be in .gitignore)

**[.planning]:**
- Purpose: GSD-generated planning documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: Yes (by this agent)
- Committed: Yes (intended for tracking architectural decisions)

---

*Structure analysis: 2026-04-07*