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


"""
Functions for loading and processing PDF documents.
"""

import os
import tempfile
from typing import List, Tuple
import streamlit as st

from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.retrievers import BM25Retriever

from app.models.embeddings import QwenEmbeddings
from app.config.constants import MAX_FILE_SIZE_MB


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded file for size and type.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File '{uploaded_file.name}' exceeds {MAX_FILE_SIZE_MB}MB limit"
    
    # Check file type
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, f"File '{uploaded_file.name}' is not a PDF"
    
    return True, ""


def load_and_chunk_pdfs(uploaded_files, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Load PDF files and chunk them into smaller pieces with validation.
    
    Args:
        uploaded_files: List of uploaded PDF files from Streamlit
        chunk_size (int): Size of each document chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[Document]: List of chunked documents
    """
    docs = []
    failed_files = []
    
    # Progress tracking
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (idx + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{total_files})")
            
            # Validate file
            is_valid, error_msg = validate_file(uploaded_file)
            if not is_valid:
                st.warning(error_msg)
                failed_files.append(uploaded_file.name)
                continue
            
            # Process file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                loader = PyMuPDFLoader(tmpfile.name)
                loaded_docs = loader.load()
                
                # Add metadata and filter empty pages
                valid_docs = []
                for doc in loaded_docs:
                    if doc.page_content.strip():  # Skip empty pages
                        doc.metadata.update({
                            "file_name": uploaded_file.name,
                            "file_size": uploaded_file.size,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap
                        })
                        valid_docs.append(doc)
                
                docs.extend(valid_docs)
            
            # Clean up temp file
            os.remove(tmpfile.name)
            
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
            failed_files.append(uploaded_file.name)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show summary
    if failed_files:
        st.warning(f"Failed to process: {', '.join(failed_files)}")
    
    if not docs:
        st.error("No valid documents were processed!")
        return []
    
    # Split documents with validation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(docs)
    
    # Filter out very short chunks
    chunked_docs = [doc for doc in chunked_docs if len(doc.page_content.strip()) > 50]
    
    st.success(f"✅ Successfully processed {len(chunked_docs)} chunks from {len(uploaded_files) - len(failed_files)} files")
    
    return chunked_docs


def build_indexes(chunks: List[Document]) -> tuple[FAISS, BM25Retriever]:
    """
    Build vector and sparse indexes from document chunks with progress tracking.
    
    Args:
        chunks (List[Document]): List of document chunks
        
    Returns:
        tuple: FAISS vector store and BM25 retriever
    """
    if not chunks:
        raise ValueError("No document chunks provided for indexing")
    
    with st.spinner("🔄 Building search indexes..."):
        # Build vector index with progress
        st.text("Creating vector embeddings...")
        embeddings = QwenEmbeddings()
        faiss_index = FAISS.from_documents(
            chunks, 
            embeddings, 
            distance_strategy=DistanceStrategy.COSINE
        )
        
        # Build sparse index
        st.text("Building keyword search index...")
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 20  # Set default retrieval count
        
        st.success(f"✅ Indexes built successfully! {len(chunks)} chunks indexed.")
    
    return faiss_index, bm25_retriever
