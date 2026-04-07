# External Integrations

**Analysis Date:** 2026-04-07

## APIs & External Services

**LLM Providers:**
- Groq - Used for LLM inference via langchain-groq integration
  - SDK/Client: langchain-groq package
  - Auth: GROQ_API_KEY environment variable
  - Models: Configurable via UI (llama 3.3, gpt-oss, etc. as mentioned in comments)

- Optional alternative LLM providers (commented out in .env):
  - OpenAI - OPENAI_API_KEY
  - Anthropic - ANTHROPIC_API_KEY  
  - Cohere - COHERE_API_KEY

**Web Search:**
- Tavily - Used for web search functionality
  - SDK/Client: tavily-python package (via langchain_community.tools.tavily_search)
  - Auth: TAVILY_API_KEY environment variable
  - Usage: TavilySearchResults tool with max_results=3

## Data Storage

**Databases:**
- FAISS (Facebook AI Similarity Search) - Vector database for document embeddings
  - Type: In-memory vector store (faiss-cpu package)
  - Persistence: Rebuilt on each session startup from processed documents
  - Client: Direct usage of FAISS library via LangChain

**File Storage:**
- Local filesystem only - Documents are processed temporarily during session
- No permanent storage of uploaded files beyond session lifetime
- PDF processing via PyMuPDF (fitz) for text extraction

**Caching:**
- Streamlit session state - Used for temporary caching during user session
- No external caching service (Redis, Memcached, etc.)

## Authentication & Identity

**Auth Provider:**
- Custom API key authentication
  - Implementation: Direct API key validation via environment variables
  - Keys validated at runtime when initializing LLM and search tools
  - No OAuth, JWT, or session-based authentication

## Monitoring & Observability

**Error Tracking:**
- None detected - Basic error handling with try/catch blocks and logging

**Logs:**
- Python logging module - Configured in app/core/agent.py
- Log format: [%(asctime)s] %(levelname)s - %(message)s
- Output: stdout via StreamHandler
- Level: INFO by default, configurable via DEBUG flag in .env

## CI/CD & Deployment

**Hosting:**
- Designed for Streamlit deployment (streamlit run streamlit_app.py or via main.py)

**CI Pipeline:**
- None detected - No CI configuration files (.github/, .gitlab-ci.yml, etc.) found

## Environment Configuration

**Required env vars:**
- GROQ_API_KEY - Required for LLM functionality
- TAVILY_API_KEY - Required for web search functionality

**Secrets location:**
- .env file (gitignored via .gitignore)
- Format: KEY=value pairs
- Never committed to repository

## Webhooks & Callbacks

**Incoming:**
- None detected - Application is client-driven via Streamlit UI

**Outgoing:**
- Groq API - HTTPS calls to Groq's inference endpoints
- Tavily API - HTTPS calls to Tavily's search endpoints
- No outbound webhooks registered

---

*Integration audit: 2026-04-07*