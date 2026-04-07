# Codebase Concerns

**Analysis Date:** 2026-04-08

## Tech Debt

**[Streamlit Session State Management]:**
- Issue: Heavy reliance on Streamlit's session state for storing critical application data (documents, indexes, agent executors) creates memory bloat and potential state inconsistency
- Files: `app/main.py`, `app/core/agent.py`, `app/core/rag_pipeline.py`
- Impact: Memory usage grows unbounded with document processing; stale state can cause errors; difficult to scale beyond single-user sessions
- Fix approach: Implement proper caching mechanism with TTL, move heavy objects to disk/database, add state cleanup routines

**[Error Handling Inconsistency]:**
- Issue: Mixed approaches to error handling - some functions return error strings, others raise exceptions, some use Streamlit warnings
- Files: Throughout codebase, particularly in `app/core/document_processor.py`, `app/core/rag_pipeline.py`
- Impact: Inconsistent user experience, difficult to trace error origins, potential unhandled exceptions
- Fix approach: Standardize on exception handling with proper logging, create custom exception types, use Streamlit's error display consistently

**[Hardcoded Configuration Values]:**
- Issue: Many configuration values are hardcoded throughout the codebase instead of being centralized
- Files: `app/core/rag_pipeline.py` (magic numbers like 60 in RRF), `app/models/embeddings.py` (batch_size=64), various UI strings
- Impact: Difficult to tune performance, requires code changes for configuration adjustments
- Fix approach: Move all tunable parameters to constants.py or configuration files, add documentation for each parameter

## Known Bugs

**[Document Processing Memory Leak]:**
- Symptoms: Temporary files not properly cleaned up in error conditions, leading to disk space accumulation
- Files: `app/core/document_processor.py` lines 77-97
- Trigger: Exception during PDF processing when temporary file creation succeeds but processing fails
- Workaround: Manual cleanup of temp directory
- Fix: Use context managers or try/finally blocks to ensure temp file deletion

**[Agent Initialization Race Condition]:**
- Symptoms: Agent executor may be accessed before full initialization in concurrent scenarios
- Files: `app/main.py` lines 165-172, `app/core/agent.py` lines 149-186
- Trigger: Rapid clicking of "Process Documents" button or concurrent user actions
- Workaround: Disable UI elements during processing
- Fix: Implement proper locking mechanism or state validation before agent usage

**[Web Search API Key Exposure]:**
- Symptoms: Tavily API key stored in session state and passed around openly
- Files: `app/core/agent.py` line 154-155, `app/main.py` line 156
- Trigger: Any inspection of session state or memory dumps
- Workaround: None - this is inherent to the current architecture
- Fix: Use Streamlit's secret management for API keys, avoid storing keys in session state

## Security Considerations

**[API Key Storage in .env]:**
- Risk: API keys committed to version control in .env file
- Files: `.env` 
- Current mitigation: .gitignore should exclude .env, but file exists in repository
- Recommendations: Remove actual keys from repository, use environment variables or secret management systems, add .env.example with placeholder values

**[Unsafe Temporary File Handling]:**
- Risk: Temporary files created with predictable names in shared temp directory
- Files: `app/core/document_processor.py` line 77
- Current mitigation: delete=False parameter with immediate cleanup
- Recommendations: Use secure temp file creation with proper permissions, ensure cleanup in all code paths

**[Lack of Input Sanitization]:**
- Risk: User input passed directly to LLMs and search tools without sanitization
- Files: `app/core/agent.py` (all tool functions), `app/core/rag_pipeline.py`
- Current mitigation: Some basic validation
- Recommendations: Implement input validation and sanitization, limit query length, filter dangerous patterns

## Performance Bottlenecks

**[Embedding Model Loading Delay]:**
- Problem: Qwen embedding model loads on first use causing noticeable delay
- Files: `app/models/embeddings.py` lines 11-19
- Cause: SentenceTransformer model loading is resource-intensive
- Improvement path: Pre-load models during app initialization, use model quantization, consider smaller embedding models

**[Document Reprocessing on Every Interaction]:**
- Problem: Document indexes rebuilt unnecessarily in some state reset scenarios
- Files: `app/main.py` lines 149-153 (file removal handling)
- Cause: Over-aggressive state clearing when files are removed
- Improvement path: Implement incremental updates to indexes, smarter state management

**[Inefficient Context Compression]:**
- Problem: Context compression algorithm inefficient for large document sets
- Files: `app/core/rag_pipeline.py` lines 75-103
- Cause: Linear processing without early termination optimization
- Improvement path: Add early break conditions, improve algorithm efficiency

## Fragile Areas

**[Session State Dependency]:**
- Files: `app/main.py`, `app/core/agent.py`, `app/core/rag_pipeline.py`
- Why fragile: Nearly all functions depend on specific session state keys being present
- Safe modification: Add comprehensive state validation at start of each function, provide clear error messages
- Test coverage: Gaps in testing state edge cases and missing keys

**[LLM Provider Lock-in]:**
- Files: `app/models/model_loader.py`, `app/core/agent.py`
- Why fragile: Direct import and usage of ChatGroq limits flexibility to switch providers
- Safe modification: Abstract LLM interface, use factory pattern for model creation
- Test coverage: Missing tests for alternative LLM configurations

## Scaling Limits

**[Single-User Session Model]:**
- Current capacity: Designed for single user per Streamlit session
- Limit: Multi-user deployment would cause state conflicts and resource exhaustion
- Scaling path: Rewrite to use proper user isolation (database-backed sessions, user-specific storage), implement request queuing

**[Memory Growth with Document Size]:**
- Current capacity: Limited by available RAM for document storage and indexes
- Limit: Large document collections (>100MB) cause performance degradation
- Scaling path: Implement disk-based vector stores (FAISS with persistence), add document pagination/indexing

**[Linear Search Scaling]:]
- Problem: Search performance degrades linearly with document count
- Files: `app/core/rag_pipeline.py` (RRF fusion, reranking)
- Limit: Noticeable delay with >50 documents
- Scaling path: Implement approximate nearest neighbor search, add caching layer for frequent queries

## Dependencies at Risk

**[SentenceTransformer Model]:**
- Risk: Reliance on external Hugging Face model "Qwen/Qwen3-Embedding-0.6B"
- Impact: Model availability changes, licensing changes, or network issues could break functionality
- Migration plan: Add local model caching, implement fallback to alternative embedding models, add model version pinning

**[LangChain Version]:]
- Risk: Heavy reliance on LangChain framework which has frequent breaking changes
- Impact: Updates could require significant code rewrites
- Migration plan: Pin specific LangChain versions, create abstraction layer for easier updates, monitor LangChain release notes

## Missing Critical Features

**[Document Deletion Support]:]
- Problem: No proper way to remove processed documents from indexes
- Blocks: Document management workflows, privacy compliance (GDPR right to be forgotten)
- Located: Missing implementation in document processor and index building

**[Progressive Document Loading]:]
- Problem: All documents must be processed before any can be queried
- Blocks: Working with large document collections
- Located: Sequential processing in document_processor.py

**[Query History and Caching]:]
- Problem: Repeated queries reprocess everything from scratch
- Blocks: Interactive exploration workflows
- Located: No caching mechanism in RAG pipeline

**[Multi-modal Support]:]
- Problem: Only supports PDF text extraction
- Blocks: Processing of scanned documents, images, other formats
- Located: PyMuPDFLoader limitation in document_processor.py

## Test Coverage Gaps

**[Error Condition Testing]:]
- What's not tested: Error handling paths, exception scenarios, edge cases
- Files: All core modules lack comprehensive error testing
- Risk: Undetected bugs in failure modes
- Priority: High

**[Concurrent Access Testing]:]
- What's not tested: Multi-user scenarios, race conditions
- Files: Session state dependent functions
- Risk: Deployment failures in production
- Priority: High

**[Performance Testing]:]
- What's not tested: Load testing, performance benchmarks
- Files: No performance test suite
- Risk: Undetected scalability issues
- Priority: Medium

**[Security Testing]:]
- What's not tested: Input validation, API key exposure, secure coding practices
- Files: Security-related functions
- Risk: Vulnerabilities in production
- Priority: High
