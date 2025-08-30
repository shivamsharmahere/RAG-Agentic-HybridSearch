"""
Agent and tool configuration for the RAG system.
"""


from typing import Dict, Any
import streamlit as st
import logging
import sys

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
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


def create_agent_and_tools(llm: ChatGroq, memory, tavily_api_key: str) -> Dict[str, Any]:
    """
    Create a simplified direct tool executor for the RAG application.
    Bypasses ReAct framework complexity since our tools already return well-formatted markdown.
    
    Args:
        llm (ChatGroq): Language model for the agent
        memory: Conversation memory for the agent
        tavily_api_key (str): API key for Tavily search
        
    Returns:
        Dict[str, Any]: Tool executor with direct tool access
    """
    
    def advanced_rag_tool(query: str) -> str:
        """
        Enhanced tool for answering questions from documents.
        Returns a markdown-formatted answer with better error handling.
        Logs all actions to the terminal.
        """
        logger.info(f"[Advanced_Document_QA] Received query: {query}", extra=None, stacklevel=1)
        sys.stdout.flush()
        try:
            # Validate query
            if not query.strip():
                logger.warning("[Advanced_Document_QA] Empty query received.", extra=None, stacklevel=1)
                sys.stdout.flush()
                return "**Please provide a valid question.**"
            # Check if indexes are available
            if not st.session_state.get("faiss_index") or not st.session_state.get("bm25_retriever"):
                logger.error("[Advanced_Document_QA] Document indexes are not built.", extra=None, stacklevel=1)
                sys.stdout.flush()
                return "**Error:** Document indexes are not built. Please upload and process documents first."
            # Check if documents exist
            if not st.session_state.get("docs"):
                logger.error("[Advanced_Document_QA] No documents available.", extra=None, stacklevel=1)
                sys.stdout.flush()
                return "**Error:** No documents available. Please upload PDF files and click 'Process Documents'."
            # Get the full result from RAG pipeline
            logger.info(f"[Advanced_Document_QA] Running RAG pipeline for query: {query}", extra=None, stacklevel=1)
            sys.stdout.flush()
            rag_result = advanced_rag_search(query, llm)
            # Store the full result in session state for UI access
            st.session_state.latest_rag_result = rag_result
            # Extract answer with confidence indicator
            answer = rag_result.get("answer", "I could not find an answer in the documents.")
            confidence = rag_result.get("confidence", 0.0)
            logger.info(f"[Advanced_Document_QA] RAG pipeline completed. Confidence: {confidence}", extra=None, stacklevel=1)
            sys.stdout.flush()
            # Add confidence indicator for user feedback
            if confidence > 0.7:
                confidence_indicator = "🟢 High confidence"
            elif confidence > 0.4:
                confidence_indicator = "🟡 Medium confidence"
            else:
                confidence_indicator = "🔴 Low confidence - consider rephrasing your question"
            # Only clean problematic characters that break ReAct, but preserve markdown
            cleaned_answer = answer.replace("```", "").replace("Action:", "Action_").replace("Observation:", "Observation_")
            # Add confidence info if not already in debug mode
            if not st.session_state.get("debug_mode", False) and confidence > 0:
                cleaned_answer += f"\n\n*{confidence_indicator}*"
            logger.info(f"[Advanced_Document_QA] Returning answer to user.", extra=None, stacklevel=1)
            sys.stdout.flush()
            return cleaned_answer
        except Exception as e:
            logger.exception(f"[Advanced_Document_QA] Error occurred while searching documents: {str(e)}", extra=None, stacklevel=1)
            sys.stdout.flush()
            error_msg = f"**Error occurred while searching documents:** {str(e)}"
            if st.session_state.get("debug_mode", False):
                error_msg += f"\n\n*Debug info: Query='{query}', Session state keys: {list(st.session_state.keys())}*"
            return error_msg
    
    def summarize_tool(_: str) -> str:
        """Enhanced tool for creating beautifully formatted document summaries with token management."""
        try:
            if not st.session_state.get("docs"): 
                return "**No documents are loaded to summarize.**"
            
            # Chunk documents to fit within token limits
            def chunk_documents(docs, max_tokens=500):  # Conservative limit
                chunked_docs = []
                for doc in docs:
                    content = doc.page_content
                    words = content.split()
                    
                    # Split into smaller chunks if content is too long
                    if len(words) > max_tokens:
                        for i in range(0, len(words), max_tokens):
                            chunk_content = " ".join(words[i:i + max_tokens])
                            chunked_doc = Document(
                                page_content=chunk_content,
                                metadata=doc.metadata.copy()
                            )
                            chunked_docs.append(chunked_doc)
                    else:
                        chunked_docs.append(doc)
                
                return chunked_docs
            
            # Create chunked documents
            chunked_docs = chunk_documents(st.session_state.docs)
            
            # Use a simpler, more direct approach to avoid token issues
            summaries = []
            doc_count = len(st.session_state.docs)
            
            # Process documents in small batches
            batch_size = 2  # Process 2 documents at a time
            
            for i in range(0, min(len(chunked_docs), 6), batch_size):  # Limit to first 6 chunks
                batch = chunked_docs[i:i + batch_size]
                
                # Create a shorter context for each batch
                batch_content = ""
                for doc in batch:
                    # Limit each document to 150 words
                    words = doc.page_content.split()[:150]
                    content = " ".join(words)
                    file_name = doc.metadata.get('file_name', 'document')
                    batch_content += f"\n**{file_name}**: {content}...\n"
                
                # Summarize this batch with a simple prompt
                prompt = f"""Summarize the following content in 2-3 key bullet points:

{batch_content}

Summary (use bullet points):"""
                
                try:
                    response = llm.invoke(prompt)
                    batch_summary = getattr(response, "content", "").strip()
                    if batch_summary:
                        summaries.append(batch_summary)
                except Exception as e:
                    summaries.append(f"**Error summarizing batch {i//batch_size + 1}:** {str(e)}")
            
            # Combine all summaries
            total_pages = sum(1 for doc in st.session_state.docs if doc.metadata.get('page') is not None)
            
            final_summary = f"""## 📚 Document Collection Summary

**📊 Collection Stats:**
- **Total Documents:** {doc_count}
- **Total Pages:** {total_pages if total_pages > 0 else 'N/A'}
- **Generated:** Recently

## 📄 Key Points from Documents

{chr(10).join(summaries)}

---

*💡 Tip: Ask specific questions about any topic mentioned above for detailed information.*
"""
            
            return final_summary
            
        except Exception as e:
            return f"**❌ Error occurred while summarizing:** {str(e)}"
    
    def web_search_tool(query: str) -> str:
        """Enhanced web search tool using Tavily API."""
        try:
            tavily_search = TavilySearchResults(max_results=3, api_key=tavily_api_key)
            results = tavily_search.invoke(query)
            
            if not results or not isinstance(results, list):
                return "**No web search results found.** Please try a different search term."
            
            # Format results as markdown
            formatted_results = "## 🌐 Web Search Results\n\n"
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content available')
                url = result.get('url', '')
                
                formatted_results += f"### [{i}] {title}\n\n"
                formatted_results += f"{content[:300]}{'...' if len(content) > 300 else ''}\n\n"
                if url:
                    formatted_results += f"🔗 [View Source]({url})\n\n"
                formatted_results += "---\n\n"
            
            # Store web results in session state for UI citations
            st.session_state.latest_web_results = results
            
            return formatted_results
            
        except Exception as e:
            return f"**Web search error:** {str(e)}"
    
    # Simple tool routing logic
    def execute_query(query: str) -> str:
        """
        Simple query router that directly calls the appropriate tool.
        Much cleaner than ReAct framework for our use case.
        """
        query_lower = query.lower().strip()
        
        # Handle greetings
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "how are you"]):
            return "Hello! I'm here to help you with your document questions. You can ask me about specific topics in your documents, request summaries, or search the web."
        
        # Handle summary requests
        if any(keyword in query_lower for keyword in ["summarize", "summary", "overview", "what's in", "what is in", "contents of"]):
            return summarize_tool(query)
        
        # Handle web search requests
        if any(keyword in query_lower for keyword in ["search", "web", "internet", "online", "latest", "current", "recent", "news"]):
            return web_search_tool(query)
        
        # Default to document Q&A
        return advanced_rag_tool(query)
    
    # Return a simple executor object
    class SimpleToolExecutor:
        def __init__(self, execute_func):
            self.execute_func = execute_func
            self.memory = memory
        
        def invoke(self, input_dict: Dict[str, str]) -> Dict[str, Any]:
            """Execute the query and return results in expected format."""
            query = input_dict.get("input", "")
            
            # Execute the appropriate tool
            result = self.execute_func(query)
            
            # Add to memory
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(result)
            
            return {
                "output": result,
                "intermediate_steps": []  # Empty since we don't use ReAct steps
            }
    
    return SimpleToolExecutor(execute_query)