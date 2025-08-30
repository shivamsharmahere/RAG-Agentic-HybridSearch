"""
UI components for the Streamlit application.
"""

import streamlit as st
from typing import Dict, Any, List


def render_sidebar() -> Dict[str, Any]:
    """
    Render the sidebar controls.
    
    Returns:
        Dict[str, Any]: Dictionary containing user inputs from the sidebar
    """
    with st.sidebar:
        st.title("🧠 Controls")
        
        if st.button("New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.session_state.docs = None
            st.session_state.faiss_index = None
            st.session_state.bm25_retriever = None
            st.session_state.agent_executor = None
            st.rerun()

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # LLM Selection
        st.subheader("1) Choose Your LLM")
        llm_selection = st.selectbox(
            "LLM",
            options=['llama-3.1-8b-instant', 'llama-3.3-70b-versatile', 'openai/gpt-oss-120b', 'mixtral-8x7b-32768'],
            index=0
        )

        # API Keys section
        st.subheader("2) API Keys")
        groq_api_key = st.text_input("GROQ API Key", type="password", value=st.session_state.get('groq_api_key', ''))
        tavily_api_key = st.text_input("Tavily API Key", type="password", help="For web search tool", 
                                       value=st.session_state.get('tavily_api_key', ''))
        
        # Document configuration section
        st.subheader("3) Upload & Configure")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        chunk_size = st.slider("Chunk Size", 200, 2000, 400, 50)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, 60, 10)
        
        # Process button
        process_clicked = st.button("Process Documents", use_container_width=True, type="primary")
        
        # Show loaded documents if any
        if st.session_state.get("processed_files"):
            st.markdown("---")
            st.write("📁 **Loaded Documents:**")
            files_to_remove = []
            for f in st.session_state.processed_files:
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.info(f"`{f}`")
                with col2:
                    if st.button("❌", key=f"remove_{f}", help=f"Remove {f}"):
                        files_to_remove.append(f)
            
            if files_to_remove:
                # This part will be handled in main.py after the sidebar is rendered
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


# Function removed as its functionality is now included in render_chat_history


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
    Generate the assistant's response and add it to chat history.
    """
    user_input = st.session_state.chat_history[-1]["content"]

    if st.session_state.get("agent_executor"):
        with st.spinner("Agent is thinking..."):
            result = st.session_state.agent_executor.invoke({"input": user_input})
            
            # Initialize default values
            answer = "I could not find an answer."
            citations = []
            tool_name = "Unknown"
            
            # Extract the output and tool information
            output = result.get('output', {})
            
            # Extract the tool name and citations from intermediate steps
            if 'intermediate_steps' in result and result['intermediate_steps']:
                # Get tool name from the first step
                tool_name = result['intermediate_steps'][0][0].tool
                
                # Extract citations from Advanced_Document_QA tool if used
                for step in result['intermediate_steps']:
                    if step[0].tool == "Advanced_Document_QA":
                        tool_output = step[1]
                        if isinstance(tool_output, dict) and 'citations' in tool_output:
                            citations = tool_output['citations']
                            break
                    
                    # Handle Tavily search results specifically
                    elif step[0].tool == "tavily_search_results_json":
                        tavily_results = step[1]
                        # Check if we got search results
                        if isinstance(tavily_results, list) and tavily_results:
                            # Create web search citations format
                            web_citations = []
                            for idx, result in enumerate(tavily_results, 1):
                                web_citations.append({
                                    "idx": idx,
                                    "file_name": result.get("title", "Web Result"),
                                    "page": "Web",
                                    "snippet": result.get("content", "No content available"),
                                    "url": result.get("url", "")
                                })
                            citations = web_citations
            
            # Get the answer from the output
            if isinstance(output, dict):
                # Output is a dictionary (from raw tool output)
                answer = output.get('answer', answer)
                if not citations and 'citations' in output:  # Only use these citations if we didn't get them from intermediate steps
                    citations = output['citations']
            else:
                # Output is a string (from agent's final answer)
                answer = output
        
        # Handle different tool scenarios
        
        # Case 1: Tavily was used but we're showing "I don't know" in the UI
        if tool_name == "tavily_search_results_json" and "don't know" in answer.lower() and output:
            answer = output  # Use the actual output from the agent
        
        # Case 2: If no tool was used or a non-document tool was used for a document query,
        # we need to force using the RAG pipeline directly
        elif tool_name == "Unknown" or (tool_name != "Advanced_Document_QA" and tool_name != "tavily_search_results_json" and 
                                      not user_input.lower().strip() in ["hi", "hello", "hey", "thanks", "thank you"]):
            # Simple list of queries that don't need RAG
            simple_queries = ["hi", "hello", "hey", "thanks", "thank you"]
            if user_input.lower().strip() not in simple_queries and "summarize" not in user_input.lower() and "summary" not in user_input.lower():
                # Direct call to RAG pipeline
                from app.core.rag_pipeline import advanced_rag_search
                if st.session_state.get("faiss_index") and st.session_state.get("bm25_retriever"):
                    with st.spinner("Getting information from documents..."):
                        rag_result = advanced_rag_search(user_input, st.session_state.llm)
                        answer = rag_result.get("answer", answer)
                        citations = rag_result.get("citations", [])
                        tool_name = "Advanced_Document_QA (forced)"
        
        # Add the processed response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer,
            "citations": citations,
            "tool_name": tool_name
        })
    else:
        # This case might not be hit if agent_executor is always checked before calling
        st.warning("Please process documents first.")
