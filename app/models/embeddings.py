"""
Custom embedding implementations.
"""

from typing import List
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings


@st.cache_resource(show_spinner="Loading Embedding Model...")
def get_qwen_embedder() -> SentenceTransformer:
    """
    Load and cache the Qwen embedding model.
    
    Returns:
        SentenceTransformer: The loaded sentence transformer model
    """
    return SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)


class QwenEmbeddings(Embeddings):
    """
    Custom Qwen embeddings class for LangChain compatibility.
    Uses the Qwen embedding model to create embeddings for documents and queries.
    """
    
    def __init__(self, batch_size: int = 64):
        """
        Initialize the QwenEmbeddings class.
        
        Args:
            batch_size (int, optional): Batch size for processing. Defaults to 64.
        """
        self.model = get_qwen_embedder()
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embeddings as float vectors
        """
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            prompt_name="document", show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query string.
        
        Args:
            text (str): Query text to embed
            
        Returns:
            List[float]: Embedding as a float vector
        """
        return self.model.encode(
            text, normalize_embeddings=True, prompt_name="query", show_progress_bar=False
        ).tolist()
