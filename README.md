# Advanced RAG Agent 🧠

A production-worthy Retrieval-Augmented Generation (RAG) based AI Chatbot that answers user queries using PDF documents as knowledge sources.

## Features

- **Modular Architecture**: Clean separation of concerns with industry-standard structure
- **Advanced RAG Pipeline**:
  - Query expansion with LLM-generated variants
  - Hybrid retrieval (dense + sparse)
  - Fusion of results using Reciprocal Rank Fusion (RRF)
  - Neural reranking with cross-encoders
  - Context compression and citation tracking
- **Multi-Tool Agent**: Combines document Q&A, summarization, and web search
- **Interactive UI**: Clean Streamlit interface with citation view
- **Flexibility**: Easy to swap out components (embeddings, models, etc.)

## Architecture

The project follows a modular architecture:

- **app/config/**: Application constants and configuration
- **app/core/**: Core RAG pipeline and agent functionality
- **app/models/**: Model implementations and loaders
- **app/ui/**: UI components and handlers
- **app/utils/**: Utility functions and helpers

## Prerequisites

- Python 3.9+
- API keys:
  - GROQ API key (for LLM access)
  - Tavily API key (for web search capabilities)

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/advanced-rag-agent.git
cd advanced-rag-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Upload PDF documents via the sidebar.

3. Configure chunk size and overlap as needed.

4. Click "Process Documents" to build the knowledge base.

5. Ask questions in the chat interface!

## Customization Options

### Using Different LLMs

To switch to a different LLM provider:

1. Install the required package (e.g., `pip install langchain-openai`)
2. Modify `models/model_loader.py` to use your preferred LLM

### Using Different Embedding Models

To use different embedding models:

1. Modify `models/embeddings.py` to use your preferred embedding model
2. Update the document processor accordingly

## Design Decisions

### Vector Database: FAISS

FAISS was chosen for its:
- Excellent performance characteristics
- In-memory storage (no extra infrastructure needed)
- Easy integration with LangChain

### LLM: Groq

Groq provides:
- High performance inference speeds
- Cost-effective API access
- Compatible with LangChain ecosystem

### Embedding Model: Qwen

Qwen embeddings offer:
- Strong performance on retrieval tasks
- Reasonable size and speed
- Built-in prompt formats for queries vs. documents

## License

This project is licensed under the MIT License - see the LICENSE file for details.
