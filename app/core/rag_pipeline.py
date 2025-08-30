"""
Core RAG pipeline functions for query processing and answer generation.
"""

from typing import List, Dict, Any, Tuple
import streamlit as st

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from app.models.model_loader import get_reranker
from app.config.constants import MAX_CONTEXT_TOKENS, RERANK_KEEP, N_QUERY_VARIANTS, MAX_CITATIONS


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
    variants = [line.strip("- ").strip() for line in getattr(resp, "content", "").splitlines() if line.strip()]
    return [query] + list(set(variants))[:N_QUERY_VARIANTS]


def rrf_fuse(list_of_lists: List[List[Document]]) -> List[Document]:
    """
    Fuse multiple document lists using Reciprocal Rank Fusion.
    
    Args:
        list_of_lists (List[List[Document]]): Multiple lists of retrieved documents
        
    Returns:
        List[Document]: Fused list of documents
    """
    scores, doc_map = {}, {}
    def doc_key(d): return f"{d.metadata.get('file_name', '')}|{d.metadata.get('page', '')}|{hash(d.page_content[:100])}"
    for docs in list_of_lists:
        for rank, d in enumerate(docs):
            key = doc_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank + 1.0)
            if key not in doc_map: doc_map[key] = d
    fused = sorted([(doc_map[k], s) for k, s in scores.items()], key=lambda x: x[1], reverse=True)
    return [d for d, _ in fused]


def rerank_docs(query: str, docs: List[Document]) -> List[Document]:
    """
    Rerank documents using a cross-encoder reranker.
    
    Args:
        query (str): User query
        docs (List[Document]): List of retrieved documents
        
    Returns:
        List[Document]: List of reranked documents
    """
    if not docs: return []
    reranker = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:RERANK_KEEP]]


def compress_context(query: str, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Compress the context to fit within token limits and prepare citations.
    
    Args:
        query (str): User query
        docs (List[Document]): List of documents to compress
        
    Returns:
        Tuple[str, List[Dict[str, Any]]]: Compressed context string and citation metadata
    """
    context_lines, citations_meta = [], []
    words_used, idx = 0, 1
    for d in docs:
        if words_used >= MAX_CONTEXT_TOKENS: break
        content = d.page_content
        words_in_content = len(content.split())
        if words_used + words_in_content > MAX_CONTEXT_TOKENS:
            allowed = max(0, MAX_CONTEXT_TOKENS - words_used)
            content = " ".join(content.split()[:allowed])
        file_name = d.metadata.get("file_name", "document")
        page = d.metadata.get("page", None)
        line = f"[{idx}] {file_name}" + (f" (p{page + 1})" if isinstance(page, int) else "") + f": {content}"
        context_lines.append(line)
        citations_meta.append({"idx": idx, "file_name": file_name, "page": page, "snippet": content[:400]})
        words_used += words_in_content
        idx += 1
        if idx > MAX_CITATIONS: break
    return "\n".join(context_lines), citations_meta


def generate_answer(llm: ChatGroq, question: str, context: str) -> str:
    """
    Generate an answer using the LLM based on the context.
    
    Args:
        llm (ChatGroq): LLM for answer generation
        question (str): User question
        context (str): Compressed context
        
    Returns:
        str: Generated answer
    """
    prompt = PromptTemplate.from_template(
        "You are a precise assistant. Answer ONLY using the provided context.\n"
        "If the context is insufficient, say: \"I don't know based on the provided documents.\"\n"
        "Cite support using bracketed numbers like [1], [2].\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ).format(question=question, context=context)
    resp = llm.invoke(prompt)
    return getattr(resp, "content", "").strip()


def advanced_rag_search(query: str, llm: ChatGroq) -> Dict[str, Any]:
    """
    Execute the complete RAG pipeline for a given query.
    
    Args:
        query (str): User query
        llm (ChatGroq): Language model for generating answers
        
    Returns:
        Dict[str, Any]: Answer and citation metadata
    """
    if not st.session_state.get("faiss_index") or not st.session_state.get("bm25_retriever"):
        return {"answer": "Error: Document indexes are not built. Please process documents first."}
    
    # 1. Query Expansion
    variants = generate_query_variants(llm, query)
    
    # 2. Hybrid Retrieval + Fusion
    candidate_lists = []
    for v in variants:
        dense_hits = st.session_state.faiss_index.as_retriever(search_kwargs={"k": 20}).get_relevant_documents(v)
        sparse_hits = st.session_state.bm25_retriever.get_relevant_documents(v)
        candidate_lists.append(rrf_fuse([dense_hits, sparse_hits]))
    fused_candidates = rrf_fuse(candidate_lists)
    
    # 3. Reranking
    reranked_docs = rerank_docs(query, fused_candidates)
    
    # 4. Context Compression & Answer Generation
    context, citations = compress_context(query, reranked_docs)
    answer = generate_answer(llm, query, context)
    
    return {"answer": answer, "citations": citations}
