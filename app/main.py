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
    Enhanced agent setup with better error handling and user feedback.
    
    Args:
        groq_api_key (str): GROQ API key
        tavily_api_key (str): Tavily API key
        uploaded_files: List of uploaded files
        chunk_size (int): Size of document chunks
        chunk_overlap (int): Overlap between chunks
        llm_selection (str): Selected LLM from Groq
    """
    setup_container = st.container()
    
    with setup_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process documents
            status_text.text("📄 Processing documents...")
            progress_bar.progress(0.2)
            
            st.session_state.docs = load_and_chunk_pdfs(uploaded_files, chunk_size, chunk_overlap)
            
            if not st.session_state.docs:
                st.error("❌ No valid documents were processed. Please check your files and try again.")
                return
            
            # Step 2: Build indexes
            status_text.text("🔍 Building search indexes...")
            progress_bar.progress(0.5)
            
            st.session_state.faiss_index, st.session_state.bm25_retriever = build_indexes(st.session_state.docs)
            
            # Step 3: Initialize LLM
            status_text.text("🤖 Initializing AI model...")
            progress_bar.progress(0.7)
            
            st.session_state.llm = get_llm(groq_api_key, model_name=llm_selection)
            
            # Step 4: Set up agent
            status_text.text("🧠 Setting up intelligent agent...")
            progress_bar.progress(0.9)
            
            st.session_state.agent_executor = create_agent_and_tools(
                st.session_state.llm, st.session_state.memory, tavily_api_key
            )
            
            # Step 5: Finalize
            progress_bar.progress(1.0)
            status_text.text("✅ Setup complete!")
            
            # Update processed files list
            st.session_state.processed_files = [f.name for f in uploaded_files]
            
            # Show summary
            doc_count = len(st.session_state.docs)
            files_count = len(uploaded_files)
            
            success_msg = f"""
            🎉 **Setup Complete!**
            
            ✅ Processed **{files_count}** files into **{doc_count}** searchable chunks  
            🤖 Using **{llm_selection}** for responses  
            🔍 Hybrid search enabled (Vector + Keyword)  
            
            **You can now ask questions about your documents!**
            """
            
            st.success(success_msg)
            
            # Clear progress indicators after a moment
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ **Setup failed:** {str(e)}")
            
            # Clear any partial state
            for key in ['docs', 'faiss_index', 'bm25_retriever', 'agent_executor', 'llm']:
                if key in st.session_state:
                    del st.session_state[key]


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
    
    # Display main content with better structure
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header(APP_TITLE)
        st.caption("💬 Ask questions about your documents • 📋 Request summaries • 🌐 Search the web")
    
    with col2:
        # Quick stats if documents are loaded
        if st.session_state.get("docs") and st.session_state.get("processed_files"):
            st.metric("📚 Documents", len(st.session_state.processed_files))
            st.metric("📄 Chunks", len(st.session_state.docs))
    
    # Show helpful examples if no documents are loaded
    if not st.session_state.get("agent_executor"):
        st.info("""
        🚀 **Get Started:**
        1. Add your API keys in the sidebar
        2. Upload PDF documents  
        3. Click "Process Documents"
        4. Start asking questions!
        
        **Example questions:**
        - "What are the main findings?"
        - "Summarize the key points"
        - "Search for recent AI developments" (web search)
        """)
    
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
