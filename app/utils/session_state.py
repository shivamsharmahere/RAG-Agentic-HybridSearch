"""
Session state initialization and management.
"""

import streamlit as st
from langchain.memory import ConversationBufferMemory


def initialize_session_state():
    """
    Initialize all session state variables with default values if they don't exist.
    """
    defaults = {
        "chat_history": [],  # Chat history for display
        "memory": ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='output'),  # LangChain memory
        "agent_executor": None,  # Agent executor
        "docs": [],  # Original document list
        "faiss_index": None,  # Vector index
        "bm25_retriever": None,  # Sparse retriever
        "processed_files": [],  # List of processed file names
        "groq_api_key": "",  # API key for GROQ
        "tavily_api_key": ""  # API key for Tavily search
    }
    
    # Initialize each key if not present
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
