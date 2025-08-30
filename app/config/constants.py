"""
Configuration constants for the RAG system.
"""

# Application configuration
APP_TITLE = "Advanced RAG Agent 🧠"
APP_VERSION = "1.0.0"

# RAG pipeline configuration
MAX_CONTEXT_TOKENS = 1800  # Maximum number of tokens to use in context window
RERANK_KEEP = 8           # Number of documents to keep after reranking
N_QUERY_VARIANTS = 3      # Number of query variants to generate
MAX_CITATIONS = 4         # Maximum number of citations to include in response

# Performance settings
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 60
MAX_FILE_SIZE_MB = 50      # Maximum file size allowed
MAX_FILES_UPLOAD = 10      # Maximum number of files that can be uploaded

# UI Constants
SPINNER_MESSAGES = {
    "processing": "🔄 Processing documents...",
    "thinking": "🧠 Agent is thinking...",
    "loading_model": "🤖 Loading AI models...",
    "searching": "🔍 Searching documents..."
}

# Error messages
ERROR_MESSAGES = {
    "no_docs": "📄 Please upload and process documents first.",
    "no_api_keys": "🔑 Please enter all required API keys.",
    "file_too_large": "📁 File size exceeds {MAX_FILE_SIZE_MB}MB limit.",
    "processing_failed": "❌ Document processing failed. Please try again."
}
