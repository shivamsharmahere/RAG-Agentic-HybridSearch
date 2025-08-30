"""
Functions for loading and processing PDF documents.
"""

import os
import tempfile
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.retrievers import BM25Retriever

from app.models.embeddings import QwenEmbeddings


def load_and_chunk_pdfs(uploaded_files, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Load PDF files and chunk them into smaller pieces.
    
    Args:
        uploaded_files: List of uploaded PDF files from Streamlit
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[Document]: List of chunked documents
    """
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            loader = PyMuPDFLoader(tmpfile.name)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["file_name"] = uploaded_file.name
            docs.extend(loaded_docs)
        os.remove(tmpfile.name)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)


def build_indexes(chunks: List[Document]) -> tuple[FAISS, BM25Retriever]:
    """
    Build vector and sparse indexes from document chunks.
    
    Args:
        chunks (List[Document]): List of document chunks
        
    Returns:
        tuple: FAISS vector store and BM25 retriever
    """
    embeddings = QwenEmbeddings()
    faiss_index = FAISS.from_documents(chunks, embeddings, distance_strategy=DistanceStrategy.COSINE)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    return faiss_index, bm25_retriever
