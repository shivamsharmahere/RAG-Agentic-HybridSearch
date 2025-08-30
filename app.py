# app.py

import os
import re
import tempfile
from typing import List, Tuple, Dict, Any, Set
import streamlit as st

# LangChain / LLM / Retrieval
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory

# Embedding + Reranker
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# =============================================================================
# Configuration
# =============================================================================

APP_TITLE = "Advanced RAG Agent 🧠"
MAX_CONTEXT_TOKENS = 1800
RERANK_KEEP = 8
N_QUERY_VARIANTS = 3
MAX_CITATIONS = 4

# =============================================================================
# Caching for Models
# =============================================================================

@st.cache_resource(show_spinner="Loading Embedding Model...")
def get_qwen_embedder() -> SentenceTransformer:
    return SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

@st.cache_resource(show_spinner="Loading Reranker Model...")
def get_reranker() -> CrossEncoder:
    return CrossEncoder("BAAI/bge-reranker-base", trust_remote_code=True)

@st.cache_resource(show_spinner=False)
def get_llm(groq_api_key: str) -> ChatGroq:
    return ChatGroq(temperature=0.1, groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

# =============================================================================
# Custom Qwen Embeddings Class
# =============================================================================

class QwenEmbeddings(Embeddings):
    def __init__(self, batch_size: int = 64):
        self.model = get_qwen_embedder()
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            prompt_name="document", show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(
            text, normalize_embeddings=True, prompt_name="query", show_progress_bar=False
        ).tolist()

# =============================================================================
# Core RAG Pipeline Functions (To be used in a tool)
# =============================================================================

def load_and_chunk_pdfs(uploaded_files, chunk_size: int, chunk_overlap: int) -> List[Document]:
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            loader = PyMuPDFLoader(tmpfile.name)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["file_name"] = uploaded_file.name
            docs.extend(loaded_docs)
        os.remove(tmpfile.name)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

def build_indexes(chunks: List[Document]) -> Tuple[FAISS, BM25Retriever]:
    embeddings = QwenEmbeddings()
    faiss_index = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    return faiss_index, bm25_retriever

def generate_query_variants(llm: ChatGroq, query: str) -> List[str]:
    prompt = PromptTemplate.from_template(
        f"Generate {N_QUERY_VARIANTS} diverse paraphrases of this query: {{q}}"
    )
    resp = llm.invoke(prompt.format(q=query))
    variants = [line.strip("- ").strip() for line in getattr(resp, "content", "").splitlines() if line.strip()]
    return [query] + list(set(variants))[:N_QUERY_VARIANTS]

def rrf_fuse(list_of_lists: List[List[Document]]) -> List[Document]:
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
    if not docs: return []
    reranker = get_reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:RERANK_KEEP]]

def compress_context(query: str, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    # Simplified context compression for brevity in the final app
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
    prompt = PromptTemplate.from_template(
        "You are a precise assistant. Answer ONLY using the provided context.\n"
        "If the context is insufficient, say: \"I don't know based on the provided documents.\"\n"
        "Cite support using bracketed numbers like [1], [2].\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ).format(question=question, context=context)
    resp = llm.invoke(prompt)
    return getattr(resp, "content", "").strip()

# =============================================================================
# Agent and Tools Setup
# =============================================================================

def create_agent_and_tools(llm: ChatGroq, memory, tavily_api_key: str):
    """Defines the agent and its tools."""

    def advanced_rag_tool(query: str) -> Dict[str, Any]:
        """The main tool for answering questions from documents."""
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

    def summarize_tool(_: str) -> str:
        """Tool for summarizing all documents."""
        if not st.session_state.get("docs"): return "No documents are loaded to summarize."
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
        return chain.run(st.session_state.docs)

    tools = [
        Tool(
            name="Advanced_Document_QA",
            func=advanced_rag_tool,
            description="Use this primary tool to answer specific questions about the content of the uploaded PDF documents. The input must be a full question from the user."
        ),
        Tool(
            name="Summarize_Documents",
            func=summarize_tool,
            description="Use this tool when the user explicitly asks for a summary of the documents. It takes a dummy string as input."
        ),
        TavilySearchResults(max_results=3, api_key=tavily_api_key)
    ]

    prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, memory=memory,
        handle_parsing_errors="I apologize, I had trouble processing that request. Please try rephrasing."
    )
    return agent_executor

# =============================================================================
# Streamlit UI
# =============================================================================

def initialize_session_state():
    defaults = {
        "chat_history": [],
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='output'),
        "agent_executor": None, "docs": [], "chunks": [], "faiss_index": None,
        "bm25_retriever": None, "processed_files": []
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧠")
    initialize_session_state()

    load_dotenv()

    with st.sidebar:
        st.title("🧠 Controls")
        st.subheader("1) API Keys")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            groq_api_key = st.text_input("GROQ API Key", type="password")
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            tavily_api_key = st.text_input("Tavily API Key", type="password", help="For web search tool")
        
        st.subheader("2) Upload & Configure")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        chunk_size = st.slider("Chunk Size", 200, 2000, 400, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 60, 10)
        
        if st.button("Process Documents", use_container_width=True, type="primary"):
            if not uploaded_files: st.warning("Please upload PDFs.")
            elif not groq_api_key or not tavily_api_key: st.warning("Please enter all API keys.")
            else:
                with st.spinner("Processing documents..."):
                    st.session_state.docs = load_and_chunk_pdfs(uploaded_files, chunk_size, chunk_overlap)
                    st.session_state.faiss_index, st.session_state.bm25_retriever = build_indexes(st.session_state.docs)
                    st.session_state.llm = get_llm(groq_api_key)
                    st.session_state.agent_executor = create_agent_and_tools(st.session_state.llm, st.session_state.memory, tavily_api_key)
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.success("Documents processed! Agent is ready.")

        if st.session_state.processed_files:
            st.markdown("---"); st.write("📁 **Loaded Documents:**")
            for f in st.session_state.processed_files: st.info(f"`{f}`")

    st.header(APP_TITLE)
    st.caption("I can answer questions about your documents, summarize them, or search the web.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "citations" in msg and msg["citations"]:
                with st.expander("View Sources"):
                    for c in msg["citations"]:
                        st.info(f"**[{c['idx']}] {c['file_name']} (p{c['page'] + 1 if isinstance(c['page'], int) else 'N/A'})**\n\n> {c['snippet']}...")

    if user_q := st.chat_input("Ask a question..."):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        if st.session_state.agent_executor:
            with st.chat_message("assistant"):
                with st.spinner("Agent is thinking..."):
                    result = st.session_state.agent_executor.invoke({"input": user_q})
                    answer, citations = result.get('output', {}), []
                    if isinstance(answer, dict):
                        citations = answer.get("citations", [])
                        answer = answer.get("answer", "I could not find an answer.")
                    st.markdown(answer)
                    if citations:
                        with st.expander("View Sources"):
                            for c in citations:
                                st.info(f"**[{c['idx']}] {c['file_name']} (p{c['page'] + 1 if isinstance(c['page'], int) else 'N/A'})**\n\n> {c['snippet']}...")
            st.session_state.chat_history.append({"role": "assistant", "content": answer, "citations": citations})
        else:
            st.warning("Please process documents first.")

if __name__ == "__main__":
    main()