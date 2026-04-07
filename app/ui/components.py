"""
UI components for the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, List

from app.config.constants import (
    MAX_FILES_UPLOAD, MAX_FILE_SIZE_MB, 
    DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
)


def render_sidebar() -> Dict[str, Any]:
    """
    Enhanced sidebar with better organization and user guidance.
    
    Returns:
        Dict[str, Any]: Dictionary containing user inputs from the sidebar
    """
    with st.sidebar:
        st.title("🧠 Controls")
        
        # Quick actions with better spacing
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🆕 New Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.processed_files = []
                st.session_state.docs = None
                st.session_state.faiss_index = None
                st.session_state.bm25_retriever = None
                st.session_state.agent_executor = None
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode", help="Show detailed processing information")
        
        st.divider()

        # LLM Selection with descriptions
        st.subheader("1️⃣ Choose Your AI Model")
        llm_options = {
            'llama-3.1-8b-instant': '⚡ Fast & Balanced',
            'llama-3.3-70b-versatile': '🧠 Most Capable',
            'openai/gpt-oss-120b': '🎯 Specialized',
            'mixtral-8x7b-32768': '📚 Long Context'
        }
        
        llm_selection = st.selectbox(
            "Model",
            options=list(llm_options.keys()),
            format_func=lambda x: f"{x} - {llm_options[x]}",
            index=0,
            help="Choose based on your needs: speed vs capability"
        )

        st.divider()

        # API Keys section with better guidance
        st.subheader("2️⃣ API Configuration")
        
        # Check if keys are loaded from env
        env_keys_loaded = bool(st.session_state.get('groq_api_key')) or bool(st.session_state.get('tavily_api_key'))
        if env_keys_loaded:
            st.success("✅ API keys loaded from environment")
        
        groq_api_key = st.text_input(
            "🤖 GROQ API Key", 
            type="password", 
            value=st.session_state.get('groq_api_key', ''),
            help="Required for AI responses. Get from: https://console.groq.com/"
        )
        
        tavily_api_key = st.text_input(
            "🌐 Tavily API Key", 
            type="password", 
            value=st.session_state.get('tavily_api_key', ''),
            help="Required for web search. Get from: https://tavily.com/"
        )
        
        st.divider()
        
        # Document configuration section with better guidance
        st.subheader("3️⃣ Document Processing")
        
        uploaded_files = st.file_uploader(
            "📁 Upload PDF Files", 
            type="pdf", 
            accept_multiple_files=True,
            help=f"Max {MAX_FILES_UPLOAD} files, {MAX_FILE_SIZE_MB}MB each"
        )
        
        # Show file count and size info
        if uploaded_files:
            total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)  # MB
            st.info(f"📊 {len(uploaded_files)} files selected ({total_size:.1f} MB total)")
        
        # Advanced settings in expander
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider(
                "Chunk Size", 
                200, 2000, 
                DEFAULT_CHUNK_SIZE, 50,
                help="Larger = more context, Smaller = more precise"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                0, 500, 
                DEFAULT_CHUNK_OVERLAP, 10,
                help="Overlap between chunks for continuity"
            )
        
        # Process button with better state indication
        can_process = uploaded_files and groq_api_key and tavily_api_key
        
        process_clicked = st.button(
            "🚀 Process Documents", 
            use_container_width=True, 
            type="primary",
            disabled=not can_process,
            help="Process documents to enable Q&A" if can_process else "Upload files and enter API keys first"
        )
        
        # Show detailed processing status
        docs = st.session_state.get("docs")
        if docs is not None and len(docs) > 0 and st.session_state.get("agent_executor"):
            doc_count = len(docs)
            file_count = len(st.session_state.get("processed_files", []))
            st.success(f"""
            ✅ System Ready!
            - {file_count} documents processed
            - {doc_count} text chunks extracted
            - Search indexes built
            - AI model initialized
            """)
        elif docs is not None and len(docs) == 0:
            st.warning("⚠️ No valid content found in uploaded documents")
        
        st.divider()
        
        # Show loaded documents with better management
        if st.session_state.get("processed_files"):
            st.subheader("� Loaded Documents")
            files_to_remove = []
            
            for f in st.session_state.processed_files:
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    st.text(f"📄 {f}")
                with col2:
                    if st.button("🗑️", key=f"remove_{f}", help=f"Remove {f}"):
                        files_to_remove.append(f)
            
            if files_to_remove:
                st.session_state.files_to_remove = files_to_remove

    return {
        "groq_api_key": groq_api_key,
        "tavily_api_key": tavily_api_key,
        "uploaded_files": uploaded_files,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "process_clicked": process_clicked,
        "llm_selection": llm_selection,
    }


def render_chat_history():
    """
    Render the chat history.
    """
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            content = msg["content"]
            
            if msg["role"] == "assistant":
                # Display the answer part
                st.markdown(content)
                
                # Display the citations in a dropdown if they exist
                if "citations" in msg and msg["citations"]:
                    with st.expander("Sources with Metadata"):
                        for c in msg["citations"]:
                            if c.get("page") == "Web" and "url" in c:
                                # Web search result with URL
                                st.info(f"**[{c['idx']}] {c['file_name']}**\n\n" + 
                                       f"> {c['snippet']}...\n\n" +
                                       f"[View Source]({c['url']})")
                            else:
                                # Document citation
                                st.info(f"**[{c['idx']}] {c['file_name']} " + 
                                       f"(p{c['page'] + 1 if isinstance(c['page'], int) else 'N/A'})**\n\n" + 
                                       f"> {c['snippet']}...")
                
                # Display the tool name in a dropdown if it exists
                if "tool_name" in msg and msg["tool_name"]:
                    with st.expander("Agent Used"):
                        st.info(msg["tool_name"])
            else:
                # User message
                st.markdown(content)


def render_chat_message(message: Dict[str, Any]):
    """
    Render a chat message with proper formatting and citations.
    
    Args:
        message (Dict[str, Any]): Message dictionary containing role, content, and optional citations
    """
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            # Render the content as markdown to support formatting
            st.markdown(message["content"])
            
            # Render citations if available
            citations = message.get("citations", [])
            if citations:
                with st.expander("📚 Sources"):
                    for c in citations:
                        if c.get("page") == "Web" and "url" in c:
                            st.info(f"**[{c['idx']}] {c['file_name']}**\n\n{c['snippet']}...\n\n[🔗 View Source]({c['url']})")
                        else:
                            page_info = f"p{c['page'] + 1}" if isinstance(c.get('page'), int) else 'N/A'
                            st.info(f"**[{c['idx']}] {c['file_name']} ({page_info})**\n\n{c['snippet']}...")


def handle_user_input(user_input: str):
    """
    Process user input and generate a response.
    
    Args:
        user_input (str): User's question
    """
    # Append user message to history and rerun to show it immediately
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.rerun()


def generate_and_render_response():
    """
    Generate the assistant's response using the simplified tool executor.
    Much cleaner without ReAct complexity.
    """
    user_input = st.session_state.chat_history[-1]["content"]

    # Enhanced state checks with better validation
    missing_components = []
    
    # Check for processed documents
    docs = st.session_state.get("docs")
    if docs is None or (isinstance(docs, list) and len(docs) == 0):
        missing_components.append("📄 Documents not processed")
    
    # Check for search indexes
    if not st.session_state.get("faiss_index") or not st.session_state.get("bm25_retriever"):
        missing_components.append("🔍 Search indexes not built")
    
    # Check for AI model
    if not st.session_state.get("agent_executor"):
        missing_components.append("🤖 AI model not initialized")
    
    if missing_components:
        error_msg = "### 🚫 System not ready\n\n" + "\n".join(f"- {msg}" for msg in missing_components)
        error_msg += "\n\nPlease upload documents and click 'Process Documents' to continue."
        st.error(error_msg)
        return

    with st.spinner("🧠 Processing your question..."):
        try:
            # Execute the query directly
            result = st.session_state.agent_executor.invoke({"input": user_input})
            
            # Get the answer
            answer = result.get('output', 'I could not process your request.')
            
            # Initialize citations and tool info
            citations = []
            tool_name = "Direct Response"
            
            # Check for document citations from RAG tool
            if st.session_state.get("latest_rag_result"):
                citations = st.session_state.latest_rag_result.get("citations", [])
                tool_name = "Document Q&A"
                # Clear after use
                del st.session_state.latest_rag_result
            
            # Check for web search citations
            elif st.session_state.get("latest_web_results"):
                web_results = st.session_state.latest_web_results
                citations = []
                for idx, result_item in enumerate(web_results, 1):
                    citations.append({
                        "idx": idx,
                        "file_name": result_item.get("title", "Web Result"),
                        "page": "Web",
                        "snippet": result_item.get("content", "No content available")[:400],
                        "url": result_item.get("url", "")
                    })
                tool_name = "Web Search"
                # Clear after use
                del st.session_state.latest_web_results
            
            # Determine tool used based on content
            if "📚 Document Collection Summary" in answer:
                tool_name = "Document Summary"
            elif "🌐 Web Search Results" in answer:
                tool_name = "Web Search"
            
            # Add the response to chat history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "citations": citations,
                "tool_name": tool_name
            })
            
        except Exception as e:
            # Error handling
            error_message = f"**Error:** {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": error_message,
                "citations": [],
                "tool_name": "Error"
            })
