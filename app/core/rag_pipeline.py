"""
Core RAG pipeline functions for query processing and answer generation.

This module implements the advanced RAG pipeline with the following components:
- Query variant generation: Creates multiple paraphrases for better retrieval
- Reciprocal Rank Fusion (RRF): Combines results from multiple retrievers
- Cross-encoder reranking: Re-orders candidate documents by relevance
- Context compression: Truncates context to fit token limits
- Answer generation: Produces markdown-formatted answers with citations
"""

from typing import List, Dict, Any, Tuple
import streamlit as st

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from app.models.model_loader import get_reranker
from app.config.constants import (
    MAX_CONTEXT_TOKENS,
    RERANK_KEEP,
    N_QUERY_VARIANTS,
    MAX_CITATIONS,
)


def generate_query_variants(llm: ChatGroq, query: str) -> List[str]:
    """
    Generate query variants using the LLM to improve retrieval.

    Args:
        llm (ChatGroq): LLM for generating variants
        query (str): Original user query

    Returns:
        List[str]: List of query variants including the original
    """
    prompt = PromptTemplate.from_template(
        f"Generate {N_QUERY_VARIANTS} diverse paraphrases of this query: {{q}}"
    )
    resp = llm.invoke(prompt.format(q=query))
    variants = [
        line.strip("- ").strip()
        for line in getattr(resp, "content", "").splitlines()
        if line.strip()
    ]
    return [query] + list(set(variants))[:N_QUERY_VARIANTS]


def rrf_fuse(list_of_lists: List[List[Document]]) -> List[Document]:
    """
    Fuse multiple document lists using Reciprocal Rank Fusion (RRF).

    RRF is a rank aggregation method that combines rankings from multiple
    retrievers without requiring score normalization. Documents appearing
    at higher ranks in any list receive higher fusion scores.

    The scoring formula is: score(d) = sum(1 / (k + rank(d))) where k=60

    Args:
        list_of_lists (List[List[Document]]): Multiple lists of retrieved documents
            from different retrievers (e.g., dense vector and sparse keyword)

    Returns:
        List[Document]: Fused and deduplicated list of documents sorted by RRF score
    """
    scores, doc_map = {}, {}

    # Create a unique key for each document based on file, page, and content hash
    # This helps identify duplicate documents across different retrievers
    def doc_key(d):
        return f"{d.metadata.get('file_name', '')}|{d.metadata.get('page', '')}|{hash(d.page_content[:100])}"

    # Accumulate RRF scores for each document across all rank positions
    for docs in list_of_lists:
        for rank, d in enumerate(docs):
            key = doc_key(d)
            # RRF scoring: constant k=60 is a common default that balances
            # the influence of top ranks vs lower ranks
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1.0)
            if key not in doc_map:
                doc_map[key] = d

    # Sort documents by fusion score (highest first) and return deduplicated list
    fused = sorted(
        [(doc_map[k], s) for k, s in scores.items()], key=lambda x: x[1], reverse=True
    )
    return [d for d, _ in fused]


def rerank_docs(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank documents using a cross-encoder reranker.

    Cross-encoder reranking provides more accurate relevance scoring than
    bi-encoder embeddings by scoring query-document pairs directly. This is
    computationally more expensive but produces better results for the
    top-k candidates from initial retrieval.

    Args:
        query (str): User query string
        docs (List[Document]): List of candidate documents from initial retrieval

    Returns:
        List[Document]: Re-ranked documents sorted by relevance score,
            truncated to RERANK_KEEP most relevant documents
    """
    # Handle edge case of empty document list
    if not docs:
        return []

    # Load the cross-encoder reranker model (cached by Streamlit)
    reranker = get_reranker()

    # Create query-document pairs for batch scoring
    # Each pair is [query_text, document_text]
    pairs = [[query, d.page_content] for d in docs]

    # Predict relevance scores for all pairs
    scores = reranker.predict(pairs)

    # Sort documents by score (descending) and keep top results
    scored = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:RERANK_KEEP]]


def compress_context(
    query: str, docs: List[Document]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Compress document content to fit within token limits while preserving citations.

    This function builds a context string by iteratively adding document content
    until the token limit (measured roughly by word count) is reached. Each document
    is formatted with a reference number and source information for citation purposes.

    Args:
        query (str): User query (unused in current implementation, reserved for future use)
        docs (List[Document]): List of ranked documents to include in context

    Returns:
        Tuple[str, List[Dict[str, Any]]]:
            - context: Formatted context string with numbered references
            - citations_meta: List of citation metadata for each included document
              Each citation contains: idx, file_name, page, snippet
    """
    context_lines, citations_meta = [], []
    words_used, idx = 0, 1

    # Iterate through documents, adding them to context until limit reached
    for d in docs:
        # Stop if we've reached token limit
        if words_used >= MAX_CONTEXT_TOKENS:
            break

        content = d.page_content
        words_in_content = len(content.split())

        # If adding this document would exceed limit, truncate it
        if words_used + words_in_content > MAX_CONTEXT_TOKENS:
            allowed = max(0, MAX_CONTEXT_TOKENS - words_used)
            content = " ".join(content.split()[:allowed])

        # Extract metadata for citation
        file_name = d.metadata.get("file_name", "document")
        page = d.metadata.get("page", None)

        # Format line with reference number and source
        # Example: [1] filename (p3): content...
        line = (
            f"[{idx}] {file_name}"
            + (f" (p{page + 1})" if isinstance(page, int) else "")
            + f": {content}"
        )
        context_lines.append(line)

        # Store citation metadata for response
        citations_meta.append(
            {"idx": idx, "file_name": file_name, "page": page, "snippet": content[:400]}
        )

        words_used += words_in_content
        idx += 1

        # Stop after reaching max citations to avoid overwhelming response
        if idx > MAX_CITATIONS:
            break

    return "\n".join(context_lines), citations_meta


def generate_answer(llm: ChatGroq, question: str, context: str) -> str:
    """
    Generate a markdown-formatted answer using the LLM based on the context.

    Args:
        llm (ChatGroq): LLM for answer generation
        question (str): User question
        context (str): Compressed context

    Returns:
        str: Generated answer with markdown formatting
    """

    prompt = PromptTemplate.from_template(
        "You are a polite, helpful, and knowledgeable AI assistant. "
        "Your task is to answer questions based ONLY on the provided context. "
        "Be clear, conversational, and engaging, similar to modern assistants like ChatGPT or Claude.\n\n"
        "Formatting rules for responses:\n"
        "- Use **bold** for important information\n"
        "- Use bullet points for lists\n"
        "- Use tables for structured data\n"
        "- Use headers (##) for sections if needed\n"
        "- Always cite sources inline as [1], [2], etc., matching the provided citations\n\n"
        "If the context is insufficient, politely say: "
        '"**I wasn’t able to find enough information in the provided documents. '
        'Could you rephrase your question or upload more documents?**"\n\n'
        "IMPORTANT: Do not use the words 'Action' or 'Observation' in your response as they interfere with parsing.\n\n"
        "Context:\n{context}\n\n"
        "User Question: {question}\n\n"
        "Answer (in a polite, interactive style):"
    ).format(question=question, context=context)

    resp = llm.invoke(prompt)
    answer = getattr(resp, "content", "").strip()

    # Only remove triple backticks and ReAct keywords that could cause parsing issues
    answer = answer.replace("```", "")
    answer = answer.replace("Action:", "Action_").replace(
        "Observation:", "Observation_"
    )

    return answer


def advanced_rag_search(query: str, llm: ChatGroq) -> Dict[str, Any]:
    """
    Perform advanced RAG search with multiple retrievers and re-ranking.
    Enhanced with better error handling and performance tracking.
    """
    import time

    start_time = time.time()

    try:
        # Validate inputs
        if not query.strip():
            return {
                "answer": "**Please provide a valid question.** A question cannot be empty.",
                "citations": [],
                "confidence": 0.0,
                "processing_time": 0.0,
            }

        # Step 1: Dual retrieval with error handling
        # Try dense vector search (FAISS) - catches semantic matches
        try:
            dense_hits = st.session_state.faiss_index.as_retriever(
                search_kwargs={"k": 20}
            ).invoke(query)
        except Exception as e:
            # Log warning but continue with empty results
            st.warning(f"Vector search (semantic matching) failed: {str(e)}")
            dense_hits = []

        # Try sparse keyword search (BM25) - catches exact matches
        try:
            if hasattr(st.session_state.bm25_retriever, "invoke"):
                sparse_hits = st.session_state.bm25_retriever.invoke(query)
            else:
                sparse_hits = st.session_state.bm25_retriever.get_relevant_documents(
                    query
                )
        except Exception as e:
            st.warning(f"Keyword search (exact matching) failed: {str(e)}")
            sparse_hits = []

        # Check if we have any results from either retriever
        if not dense_hits and not sparse_hits:
            return {
                "answer": "**No relevant documents found.**\n\nPossible reasons:\n- Your question doesn't match the content in uploaded documents\n- Documents may not have been processed correctly\n\nTips:\n- Try rephrasing your question\n- Check that documents were processed successfully",
                "citations": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
            }

        # Step 2: Fusion and reranking
        # Combine results from both retrievers using RRF
        candidate_lists = rrf_fuse([dense_hits, sparse_hits])

        if not candidate_lists:
            return {
                "answer": "**No relevant content found after combining search results.**\n\nTry a different question or upload more relevant documents.",
                "citations": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
            }

        # Step 3: Rerank with fallback
        # Re-order candidates using cross-encoder for better relevance
        try:
            reranked_docs = rerank_docs(query, candidate_lists)
        except Exception as e:
            st.warning(f"Reranking failed, using original order: {str(e)}")
            reranked_docs = candidate_lists[:RERANK_KEEP]

        # Step 4: Generate answer
        # Compress context to fit within token limits
        context, citations = compress_context(query, reranked_docs)

        if not context.strip():
            return {
                "answer": "**Not enough context found to answer your question.**\n\nThe retrieved documents don't contain sufficient relevant content. Try asking about a different topic or upload more documents.",
                "citations": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
            }

        answer = generate_answer(llm, query, context)

        # Calculate confidence based on number of relevant docs and reranking scores
        # More retrieved and reranked documents = higher confidence
        confidence = min(0.9, len(reranked_docs) / RERANK_KEEP * 0.8 + 0.1)

        processing_time = time.time() - start_time

        # Add performance info in debug mode
        if st.session_state.get("debug_mode", False):
            answer += f"\n\n*🔍 Retrieved {len(candidate_lists)} documents, reranked top {len(reranked_docs)}, processed in {processing_time:.2f}s*"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "processing_time": processing_time,
            "retrieved_docs": len(candidate_lists),
            "reranked_docs": len(reranked_docs),
        }

    except Exception as e:
        return {
            "answer": f"**An unexpected error occurred during search:**\n\n{str(e)}\n\nPlease try rephrasing your question. If the problem persists, try refreshing the page and processing documents again.",
            "citations": [],
            "confidence": 0.0,
            "processing_time": time.time() - start_time,
        }
