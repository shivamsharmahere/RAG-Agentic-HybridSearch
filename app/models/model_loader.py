"""
Model loading and caching functions.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_groq import ChatGroq


@st.cache_resource(show_spinner="Loading Reranker Model...")
def get_reranker() -> CrossEncoder:
    """
    Load and cache the reranker model.
    
    Returns:
        CrossEncoder: The loaded cross-encoder model for reranking
    """
    return CrossEncoder("BAAI/bge-reranker-base", trust_remote_code=True)


@st.cache_resource(show_spinner=False)
def get_llm(groq_api_key: str, model_name: str = "llama3-8b-8192") -> ChatGroq:
    """
    Initialize the LLM with Groq API.
    
    Args:
        groq_api_key (str): API key for Groq
        model_name (str): The name of the model to use.
        
    Returns:
        ChatGroq: Initialized LLM model
    """
    return ChatGroq(
        temperature=0.1, 
        groq_api_key=groq_api_key, 
        model_name=model_name
    )
