"""
Session state initialization and management.

This module handles initialization of Streamlit session state variables
that need to persist across user interactions within a single session.
"""

import streamlit as st
from langchain.memory import ConversationBufferMemory


def initialize_session_state():
    """
    Initialize all session state variables with default values if they don't exist.

    Session state persists data across Streamlit re-runs within the same session.
    These variables store:
    - Chat history and memory
    - Document processing state
    - Search indexes
    - API keys
    """
    # Define default values for all session state variables
    defaults = {
        "chat_history": [],  # List of message dicts: [{"role": "user/assistant", "content": "..."}]
        "memory": ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="output"
        ),  # LangChain memory for agent conversation context
        "agent_executor": None,  # ReAct agent executor with tools
        "docs": None,  # List of processed document chunks (LangChain Document objects)
        "faiss_index": None,  # FAISS vector store for semantic search
        "bm25_retriever": None,  # BM25 retriever for keyword search
        "processed_files": [],  # List of successfully processed file names
        "groq_api_key": "",  # GROQ API key for LLM
        "tavily_api_key": "",  # Tavily API key for web search
    }

    # Initialize each key only if not already present
    # This preserves existing state on subsequent re-runs
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_api_keys(groq_api_key: str, tavily_api_key: str):
    """
    Update API keys in the session state.

    Args:
        groq_api_key (str): GROQ API key
        tavily_api_key (str): Tavily API key
    """
    if groq_api_key:
        st.session_state["groq_api_key"] = groq_api_key
    if tavily_api_key:
        st.session_state["tavily_api_key"] = tavily_api_key
