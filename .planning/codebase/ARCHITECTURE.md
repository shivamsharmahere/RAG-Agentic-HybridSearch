# Architecture

**Analysis Date:** 2026-04-07

## Pattern Overview

**Overall:** Modular Layered Architecture with Separation of Concerns

**Key Characteristics:**
- Clean separation between UI, business logic, data processing, and model layers
- Component-based UI architecture using Streamlit
- Pipeline-based data flow for document processing and retrieval
- Dependency injection for LLM and embedding models
- Event-driven user interactions through Streamlit's reactive paradigm

## Layers

**[Presentation Layer]:**
- Purpose: Handles user interface and user interactions
- Location: `app/ui/` and `app/main.py`
- Contains: Streamlit components, layout rendering, event handlers
- Depends on: Application layer for business logic
- Used by: End users through web browser

**[Application Layer]:**
- Purpose: Orchestrates business logic and coordinates between layers
- Location: `app/core/` and `app/main.py`
- Contains: Agent creation, tool definitions, RAG pipeline, session state management
- Depends on: Domain layer for data models, Infrastructure layer for models and storage
- Used by: Presentation layer

**[Domain Layer]:**
- Purpose: Contains core business logic and data processing algorithms
- Location: `app/core/document_processor.py`, `app/core/rag_pipeline.py`, `app/core/agent.py`
- Contains: Document loading/chunking, indexing, retrieval, reranking, answer generation
- Depends on: Infrastructure layer for model access
- Used by: Application layer

**[Infrastructure Layer]:**
- Purpose: Provides external services and technical capabilities
- Location: `app/models/`, `app/config/`, `app/utils/`
- Contains: LLM loaders, embedding models, configuration constants, session state management
- Depends on: External APIs (Groq, Tavily, HuggingFace)
- Used by: All layers

## Data Flow

**[Document Processing Flow]:**

1. User uploads PDF files through Streamlit UI (`app/ui/components.py`)
2. Files validated and processed in `app/core/document_processor.py`:
   - Loaded with PyMuPDFLoader
   - Chunked with RecursiveCharacterTextSplitter
   - Metadata added (filename, size, chunk parameters)
3. Indexes built in `app/core/document_processor.py`:
   - FAISS vector store with Qwen embeddings
   - BM25 sparse retriever
4. Indexes stored in Streamlit session state (`app/utils/session_state.py`)

**[Query Processing Flow]:**

1. User submits question through chat interface (`app/main.py`)
2. Agent executor routes query to appropriate tool (`app/core/agent.py`):
   - Advanced RAG tool for document questions
   - Summarization tool for document summaries
   - Web search tool for external information
3. Advanced RAG pipeline execution (`app/core/rag_pipeline.py`):
   - Query variant generation for improved recall
   - Dual retrieval (vector + keyword) with RRF fusion
   - Reranking with cross-encoder model
   - Context compression to fit token limits
   - Answer generation with source citation
4. Results returned to UI with metadata for display

**[State Management]:**

- Streamlit session state stores all transient data:
  - Chat history (`chat_history`)
  - Processed documents (`docs`)
  - Search indexes (`faiss_index`, `bm25_retriever`)
  - API keys (`groq_api_key`, `tavily_api_key`)
  - Agent executor (`agent_executor`)
  - Conversation memory (`memory`)
- State initialized in `app/utils/session_state.py`
- Updated through various functions across modules

## Key Abstractions

**[RAG Agent]:**
- Purpose: Encapsulates the intelligent question-answering system
- Examples: `app/core/agent.py` (create_agent_and_tools function)
- Pattern: ReAct (Reasoning + Acting) agent with custom tools
- Responsibilities: Tool orchestration, memory management, error handling

**[Document Processor]:**
- Purpose: Handles ingestion and preparation of documents for retrieval
- Examples: `app/core/document_processor.py`
- Pattern: Pipeline with validation, loading, chunking, and indexing stages
- Responsibilities: File validation, text extraction, chunking, index building

**[RAG Pipeline]:**
- Purpose: Implements advanced retrieval-augmented generation logic
- Examples: `app/core/rag_pipeline.py`
- Pattern: Multi-stage retrieval with query expansion, fusion, reranking, and compression
- Responsibilities: Query processing, document retrieval, context preparation, answer generation

**[Model Loader]:**
- Purpose: Abstracts model initialization and caching
- Examples: `app/models/model_loader.py`, `app/models/embeddings.py`
- Pattern: Singleton/caching with Streamlit's @st.cache_resource decorator
- Responsibilities: LLM initialization, embedding model loading, reranker model loading

## Entry Points

**[Main Application]:**
- Location: `app/main.py`
- Triggers: Streamlit server execution (`streamlit run app/main.py`)
- Responsibilities:
  - Page configuration and layout
  - Session state initialization
  - Environment variable loading
  - Sidebar rendering and input handling
  - Document processing coordination
  - Chat interface rendering
  - User input processing and response generation

**[Streamlit Components]:**
- Location: `app/ui/components.py`
- Triggers: Called from `app/main.py`
- Responsibilities:
  - Sidebar rendering with controls and configuration
  - Chat history display with citations
  - User input handling
  - Response rendering with tool attribution

## Error Handling

**Strategy:** Defensive error handling with user-friendly messages and logging

**Patterns:**
- Try-catch blocks with specific exception handling in core functions
- Streamlit warning/error display for user feedback (`st.warning()`, `st.error()`)
- Logging setup in `app/core/agent.py` for debugging and monitoring
- Graceful degradation when components fail (e.g., fallback to original order when reranking fails)
- Validation functions for inputs (file validation in `document_processor.py`)
- Session state cleanup on errors to prevent inconsistent state

## Cross-Cutting Concerns

**Logging:** Standard Python logging module configured in `app/core/agent.py` with INFO level and formatted output to stdout

**Validation:** 
- File validation in `app/core/document_processor.py` (size and type checking)
- Input validation in RAG pipeline functions (empty query checks)
- API key validation in UI before processing

**Authentication:** 
- API key-based authentication for external services (Groq, Tavily)
- Keys stored in session state and/or environment variables
- No built-in user authentication system (designed for single-user/local use)

**Configuration:**
- Centralized constants in `app/config/constants.py`
- Environment variables for API keys (.env file)
- Streamlit secrets management compatible

**Performance:**
- Model caching with Streamlit's @st.cache_resource decorator
- Batch processing in embedding generation
- Progressive UI updates with progress bars and status text
- Efficient retrieval with FAISS and BM25 hybrid search

---

*Architecture analysis: 2026-04-07*