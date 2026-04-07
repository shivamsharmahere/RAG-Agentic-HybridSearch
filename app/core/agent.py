"""
Production-ready RAG Agent with ReAct architecture.

This module implements an intelligent agent that combines:
- Advanced RAG (Retrieval-Augmented Generation) for document Q&A
- Document summarization
- Web search capabilities

The agent uses LangChain's ReAct framework to intelligently route user queries
to the appropriate tool based on the query content.

Outputs are markdown-compatible and suitable for Streamlit UI.
"""

import sys
import logging
import re
import streamlit as st
from typing import Dict, Any, Optional, List
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from app.core.rag_pipeline import advanced_rag_search


# --- Logging setup ---
logger = logging.getLogger("rag_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)


# --- Tool 1: Advanced RAG Tool ---
def advanced_rag_tool(query: str) -> str:
    """
    Enhanced tool for answering questions from documents.
    Returns markdown-formatted answer with confidence indicators.
    """
    logger.info(f"[Advanced_Document_QA] Received query: {query}")
    try:
        if not query.strip():
            return "**Please provide a valid question.**"

        if not st.session_state.get("faiss_index") or not st.session_state.get(
            "bm25_retriever"
        ):
            return "**Error:** Document indexes are not built. Please upload and process documents first."

        if not st.session_state.get("docs"):
            return "**Error:** No documents available. Please upload PDF files and click 'Process Documents'."

        # Run full RAG pipeline
        rag_result = advanced_rag_search(query, st.session_state.llm)
        st.session_state.latest_rag_result = rag_result

        answer = rag_result.get(
            "answer", "I could not find an answer in the documents."
        )
        confidence = rag_result.get("confidence", 0.0)

        # Confidence indicator - maps confidence score to human-readable label
        # Thresholds: >0.7 = High, >0.4 = Medium, <=0.4 = Low
        if confidence > 0.7:
            confidence_indicator = "🟢 High confidence"
        elif confidence > 0.4:
            confidence_indicator = "🟡 Medium confidence"
        else:
            confidence_indicator = (
                "🔴 Low confidence — consider rephrasing your question"
            )

        # Sanitize output for ReAct agent parsing
        # ReAct uses "Action:" and "Observation:" as control keywords,
        # so we prefix them with underscore to prevent parsing conflicts
        # Also remove code blocks (```) which can break agent output
        cleaned_answer = re.sub(
            r"^```.*$", "", answer, flags=re.MULTILINE
        )  # Remove code block markers
        cleaned_answer = re.sub(
            r"^Action:\s", "Action_: ", cleaned_answer, flags=re.MULTILINE
        )  # Escape Action: at line start
        cleaned_answer = re.sub(
            r"^Observation:\s", "Observation_: ", cleaned_answer, flags=re.MULTILINE
        )  # Escape Observation: at line start

        if confidence > 0:
            cleaned_answer += f"\n\n*{confidence_indicator}*"

        return cleaned_answer

    except Exception as e:
        # Log full exception traceback for debugging
        logger.exception(f"[Advanced_Document_QA] Error: {str(e)}")

        # Build user-friendly error message
        error_msg = f"**An error occurred while searching your documents:**\n\n{str(e)}"

        # Include debug information if debug mode is enabled
        # This helps users and developers diagnose issues
        if st.session_state.get("debug_mode", False):
            error_msg += f"\n\n*Debug info: Query='{query}', Available session keys={list(st.session_state.keys())}*"

        return error_msg


# --- Tool 2: Summarization Tool ---
def summarize_tool(_: str) -> str:
    """
    Summarizes loaded PDF documents into key points.

    Uses the LLM to generate concise bullet-point summaries of document content.
    Limits processing to first 5 documents and 200 words per document to avoid
    token limit issues with the LLM.

    Args:
        _: str - Unused parameter required by LangChain Tool interface

    Returns:
        str: Markdown-formatted summary with sections for each document
    """
    try:
        # Check if documents are loaded in session state
        if not st.session_state.get("docs"):
            return "**No documents are loaded to summarize.**"

        docs = st.session_state.docs

        # Process limited number of documents to avoid token limits
        # Each document is truncated to 200 words to further reduce size
        summaries = []
        for i, doc in enumerate(docs[:5], start=1):  # Limit to first 5 docs
            # Extract first 200 words from document content
            content = " ".join(doc.page_content.split()[:200])

            # Build prompt for LLM summarization
            prompt = f"""Summarize the following document in 3 concise bullet points:

Document excerpt:
{content}

Summary (markdown bullets):"""

            # Invoke LLM to generate summary
            response = st.session_state.llm.invoke(prompt)
            batch_summary = getattr(response, "content", "").strip()
            if batch_summary:
                summaries.append(f"### 📄 Document {i}\n{batch_summary}")

        # Combine all summaries into final response
        return "## 📚 Document Summaries\n\n" + "\n\n".join(summaries)

    except Exception as e:
        return f"**❌ An error occurred while generating summaries:** {str(e)}"


# --- Tool 3: Web Search Tool ---
def web_search_tool(query: str) -> str:
    """
    Performs a web search using Tavily and formats results in markdown.

    Uses the Tavily search API to find recent or external information.
    Returns up to 3 results formatted as markdown with titles, content snippets,
    and links to sources.

    Args:
        query (str): Search query string

    Returns:
        str: Markdown-formatted search results with source links
    """
    try:
        # Initialize Tavily search with API key from session state
        # Max 3 results to keep response concise
        tavily_search = TavilySearchResults(
            max_results=3, api_key=st.session_state.get("tavily_api_key")
        )
        results = tavily_search.invoke(query)

        # Validate results returned from API
        if not results or not isinstance(results, list):
            return "**No web search results found.** Please try a different search term or check your API key."

        # Format results as markdown with numbered entries
        formatted_results = "## 🌐 Web Search Results\n\n"
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content available")
            url = result.get("url", "")

            # Truncate long content snippets to ~300 characters
            formatted_results += f"### [{i}] {title}\n\n"
            formatted_results += (
                f"{content[:300]}{'...' if len(content) > 300 else ''}\n\n"
            )
            if url:
                formatted_results += f"🔗 [View Source]({url})\n\n"
            formatted_results += "---\n\n"

        # Store results in session state for citation tracking
        st.session_state.latest_web_results = results
        return formatted_results

    except Exception as e:
        return f"**Web search error:** {str(e)}\n\nPlease verify your Tavily API key is valid."


# --- Agent Creation ---
def create_agent_and_tools(
    llm: ChatGroq, memory: ConversationBufferMemory, tavily_api_key: str
):
    """
    Creates a ReAct agent with tools for Advanced RAG, summarization, and web search.
    Ensures outputs are markdown-compatible.
    """
    st.session_state.llm = llm
    st.session_state.tavily_api_key = tavily_api_key

    tools = [
        Tool(
            name="Advanced_Document_QA",
            func=advanced_rag_tool,
            description="Answer questions about uploaded PDF documents.",
        ),
        Tool(
            name="Summarize_Documents",
            func=summarize_tool,
            description="Summarize the content of the uploaded documents.",
        ),
        Tool(
            name="Web_Search",
            func=web_search_tool,
            description="Search the web for recent or external information.",
        ),
    ]

    # Use LangChain Hub’s standard ReAct chat prompt
    prompt = hub.pull("hwchase17/react-chat")

    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors="⚠️ I had trouble processing that request. Please try rephrasing.",
    )
