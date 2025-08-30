"""
Main entry point for the Streamlit application.
"""

import os
from dotenv import load_dotenv
import streamlit as st

from app.config.constants import APP_TITLE
from app.core.document_processor import load_and_chunk_pdfs, build_indexes
from app.core.agent import create_agent_and_tools
from app.models.model_loader import get_llm
from app.ui.components import render_sidebar, render_chat_history, handle_user_input, generate_and_render_response
from app.utils.session_state import initialize_session_state, update_api_keys


def setup_agent(groq_api_key: str, tavily_api_key: str, uploaded_files, chunk_size: int, chunk_overlap: int, llm_selection: str):
    """
    Set up the RAG agent with the given configuration.
    
    Args:
        groq_api_key (str): GROQ API key
        tavily_api_key (str): Tavily API key
        uploaded_files: List of uploaded files
        chunk_size (int): Size of document chunks
        chunk_overlap (int): Overlap between chunks
        llm_selection (str): Selected LLM from Groq
    """
    with st.spinner("Processing documents..."):
        # Load and process documents
        st.session_state.docs = load_and_chunk_pdfs(uploaded_files, chunk_size, chunk_overlap)
        
        # Build vector and sparse indexes
        st.session_state.faiss_index, st.session_state.bm25_retriever = build_indexes(st.session_state.docs)
        
        # Initialize LLM
        st.session_state.llm = get_llm(groq_api_key, model_name=llm_selection)
        
        # Set up agent
        st.session_state.agent_executor = create_agent_and_tools(
            st.session_state.llm, st.session_state.memory, tavily_api_key
        )
        
        # Update processed files list
        st.session_state.processed_files = [f.name for f in uploaded_files]
        
        st.success("Documents processed! Agent is ready.")


def main():
    """
    Main function to run the Streamlit application.
    """
    # Configure page settings
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🧠")
    
    # Initialize session state
    initialize_session_state()
    
    # Load environment variables
    load_dotenv()
    
    # Get environment variables if available
    env_groq_api_key = os.getenv("GROQ_API_KEY")
    env_tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    # Update session state with environment variables if available
    update_api_keys(env_groq_api_key or "", env_tavily_api_key or "")
    
    # Render sidebar and get user inputs
    sidebar_inputs = render_sidebar()
    
    # Handle file removal
    if "files_to_remove" in st.session_state and st.session_state.files_to_remove:
        for f in st.session_state.files_to_remove:
            st.session_state.processed_files.remove(f)
        
        # Clear the removal list and rerun
        st.session_state.files_to_remove = []
        
        # In a real app, you'd also update docs and indexes here.
        # For simplicity, we just remove from the list. A full refresh is needed for full effect.
        # A simple way is to clear everything related to processed docs
        st.session_state.docs = None
        st.session_state.faiss_index = None
        st.session_state.bm25_retriever = None
        st.session_state.agent_executor = None
        st.rerun()

    # Update API keys in session state
    update_api_keys(sidebar_inputs["groq_api_key"], sidebar_inputs["tavily_api_key"])
    
    # Process documents if requested
    if sidebar_inputs["process_clicked"]:
        if not sidebar_inputs["uploaded_files"]:
            st.warning("Please upload PDFs.")
        elif not st.session_state.groq_api_key or not st.session_state.tavily_api_key:
            st.warning("Please enter all API keys.")
        else:
            setup_agent(
                st.session_state.groq_api_key,
                st.session_state.tavily_api_key,
                sidebar_inputs["uploaded_files"],
                sidebar_inputs["chunk_size"],
                sidebar_inputs["chunk_overlap"],
                sidebar_inputs["llm_selection"]
            )
    
    # Display main content
    st.header(APP_TITLE)
    st.caption("I can answer questions about your documents, summarize them, or search the web.")
    
    # Display chat history
    render_chat_history()
    
    # Handle user input
    if user_input := st.chat_input("Ask a question..."):
        handle_user_input(user_input)

    # Generate response if the last message is from the user
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        generate_and_render_response()
        st.rerun()


if __name__ == "__main__":
    main()
