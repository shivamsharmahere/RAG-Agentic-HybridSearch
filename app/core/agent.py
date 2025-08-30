"""
Agent and tool configuration for the RAG system.
"""

from typing import Dict, Any
import streamlit as st

from langchain.chains.summarize import load_summarize_chain
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_groq import ChatGroq
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory

from app.core.rag_pipeline import advanced_rag_search


def create_agent_and_tools(llm: ChatGroq, memory, tavily_api_key: str) -> AgentExecutor:
    """
    Create an agent with tools for the RAG application.
    
    Args:
        llm (ChatGroq): Language model for the agent
        memory: Conversation memory for the agent
        tavily_api_key (str): API key for Tavily search
        
    Returns:
        AgentExecutor: Configured agent executor
    """
    
    def advanced_rag_tool(query: str) -> Dict[str, Any]:
        """
        The main tool for answering questions from documents.
        Always returns a dictionary with 'answer' and 'citations' keys.
        """
        # Direct access to the RAG pipeline
        if not st.session_state.get("faiss_index") or not st.session_state.get("bm25_retriever"):
            return {"answer": "Error: Document indexes are not built. Please process documents first."}
        
        # We're returning the raw dictionary from advanced_rag_search
        # This ensures citations are always available to be displayed in the UI
        return advanced_rag_search(query, llm)
    
    def summarize_tool(_: str) -> str:
        """Tool for summarizing all documents."""
        if not st.session_state.get("docs"): 
            return "No documents are loaded to summarize."
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
        return chain.run(st.session_state.docs)
    
    # Define the tools
    tools = [
        Tool(
            name="Advanced_Document_QA",
            func=advanced_rag_tool,
            description=(
                "THIS IS THE MANDATORY TOOL for answering ANY question that might be related to documents. "
                "The input must be the user's full question. "
                "This tool returns a dictionary with 'answer' and 'citations'. "
                "After using this tool, provide the exact 'answer' field as your Final Answer. "
                "You MUST use this tool for ALL queries except simple greetings or explicit requests for other tools. "
                "If unsure whether a question relates to documents, ALWAYS use this tool."
            )
        ),
        Tool(
            name="Summarize_Documents",
            func=summarize_tool,
            description="Use this tool when the user explicitly asks for a summary of the documents. It takes a dummy string as input."
        ),
        TavilySearchResults(max_results=3, api_key=tavily_api_key)
    ]
    
    # Create the agent with a timeout limit to prevent long-running operations
    prompt = hub.pull("hwchase17/react-chat")
    
    # Create a custom agent prompt that forces using the Advanced_Document_QA tool
    custom_template = (
        "You are an AI assistant specialized in answering questions about documents.\n\n"
        "CRITICAL INSTRUCTION: For ANY user question that could possibly be about documents or their content, "
        "you MUST use the Advanced_Document_QA tool to get the answer. Do not try to answer these questions "
        "from memory, even if you think you know the answer.\n\n"
        "Only for these specific types of questions, you can answer directly without tools:\n"
        "1. Simple greetings like 'hello' or 'hi'\n"
        "2. Questions about yourself\n"
        "3. Explicit requests to use a different tool\n\n"
        "For EVERYTHING ELSE, you MUST use the Advanced_Document_QA tool, NO EXCEPTIONS.\n"
        "Even if you think you know the answer or aren't sure if the question relates to documents, "
        "ALWAYS use the Advanced_Document_QA tool. This is a strict requirement.\n\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "{agent_scratchpad}"
    )
    
    # Replace the prompt template
    prompt = prompt.partial(template=custom_template)
    
    agent = create_react_agent(llm, tools, prompt)
    
    # Define a function to pre-process user input and force using the RAG tool
    def intercept_query(inputs):
        """Intercept user queries and force using RAG for all non-trivial queries"""
        query = inputs.get("input", "")
        
        # Very short list of simple queries that don't need RAG
        simple_queries = ["hi", "hello", "hey", "thanks", "thank you"]
        if query.lower().strip() in simple_queries:
            return inputs
        
        # For summarization requests, don't force the RAG tool
        if query.lower().strip().startswith("summarize") or "summary" in query.lower():
            return inputs
        
        # For everything else, ALWAYS force using the Advanced_Document_QA tool
        # This ensures that all content questions go through RAG
        return {"input": f"Use the Advanced_Document_QA tool to answer this question: {query}"}
    
    # Set up the agent with intercept function
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors="I apologize, I had trouble processing that request. Please try rephrasing.",
        max_iterations=3,  # Limit iterations to prevent long-running loops
        max_execution_time=30,  # Set a 30-second timeout
        input_processor=intercept_query  # Add the interceptor to force RAG tool usage
    )
    
    return agent_executor
