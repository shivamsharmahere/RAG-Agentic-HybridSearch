"""
Configuration constants for the RAG system.
"""

# Application configuration
APP_TITLE = "Advanced RAG Agent 🧠"

# RAG pipeline configuration
MAX_CONTEXT_TOKENS = 1800  # Maximum number of tokens to use in context window
RERANK_KEEP = 8           # Number of documents to keep after reranking
N_QUERY_VARIANTS = 3      # Number of query variants to generate
MAX_CITATIONS = 4         # Maximum number of citations to include in response
