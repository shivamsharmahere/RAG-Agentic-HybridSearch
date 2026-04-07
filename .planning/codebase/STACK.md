# Technology Stack

**Analysis Date:** 2026-04-07

## Languages

**Primary:**
- Python 3.9+ - Primary language for the entire application

## Runtime

**Environment:**
- CPython 3.9+

**Package Manager:**
- pip (inferred from requirements.txt and pyproject.toml)
- Lockfile: requirements.txt (present)

## Frameworks

**Core:**
- Streamlit>=1.30.0 - Web application framework for UI
- LangChain>=0.1.0 - LLM orchestration framework
- LangChain Community>=0.0.13 - Community integrations for LangChain
- LangChain Groq>=0.1.0 - Groq-specific LangChain integration
- LangChain Core>=0.1.0 - Core LangChain abstractions

**Testing:**
- pytest>=7.0.0 - Testing framework (dev dependency)

**Build/Dev:**
- black>=23.0.0 - Code formatter (dev dependency)
- isort>=5.12.0 - Import sorter (dev dependency)
- mypy>=1.0.0 - Type checker (dev dependency)

## Key Dependencies

**Critical:**
- langchain-groq - Enables integration with Groq LLMs
- faiss-cpu>=1.7.4 - Vector similarity search for document retrieval
- sentence-transformers>=2.3.1 - Embedding models for text vectorization
- pymupdf>=1.23.8 - PDF processing library
- tavily-python>=0.2.8 - Web search API client
- python-dotenv>=1.0.0 - Environment variable management
- streamlit>=1.30.0 - Interactive web UI framework

**Infrastructure:**
- numpy==1.26.4 - Numerical computing foundation
- pandas==2.2.3 - Data manipulation and analysis
- scipy==1.13.1 - Scientific computing algorithms

## Configuration

**Environment:**
- Configured via .env file (not committed) with API keys
- Key configs: GROQ_API_KEY, TAVILY_API_KEY
- Optional: OPENAI_API_KEY, ANTHROPIC_API_KEY, COHERE_API_KEY
- Application settings: MAX_CONTEXT_TOKENS, RERANK_KEEP, etc.

**Build:**
- pyproject.toml - Build configuration and dependencies
- .tool configurations: black, isort, mypy

## Platform Requirements

**Development:**
- Python 3.9 or higher
- pip package manager

**Production:**
- Any platform supporting Python 3.9+
- Streamlit-compatible deployment (Streamlit Cloud, Docker, etc.)

---

*Stack analysis: 2026-04-07*