

---

## 🧩 Customization & Extension

### Swap LLMs
- Install your preferred LLM package (e.g., `pip install langchain-openai`)
- Edit `app/models/model_loader.py` to use your LLM

### Change Embeddings
- Edit `app/models/embeddings.py` to use a different embedding model
- Update the document processor if needed

### Add New Tools
- Add a new function in `app/core/agent.py`
- Register it in the agent's tool routing logic

---

## � Configuration

- **Chunk Size/Overlap**: Adjustable in the sidebar
- **API Keys**: Set via `.env` or sidebar
- **Constants**: See `app/config/constants.py`
- **Logging**: All agent actions and errors are logged to the terminal

---

## 🛡️ Troubleshooting & FAQ

- **No answer returned?**
  - Ensure documents are processed and API keys are set
  - Check terminal logs for errors
- **Web search not working?**
  - Verify your Tavily API key
- **Slow performance?**
  - Use smaller PDFs or reduce chunk size
- **Logs not visible?**
  - Make sure you start Streamlit from a terminal, not from an IDE run button

---

## 🤝 Contributing

1. Fork the repo and create your branch (`git checkout -b feature/your-feature`)
2. Commit your changes (`git commit -am 'Add new feature'`)
3. Push to the branch (`git push origin feature/your-feature`)
4. Open a Pull Request

Please follow [PEP8](https://peps.python.org/pep-0008/) and use `black`/`isort` for formatting.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## � Contact

- Project Maintainer: [RAG Developer](mailto:example@example.com)
- [GitHub Issues](https://github.com/shivamsharmahere/RAG-Agentic-HybridSearch/issues) for bug reports and feature requests
  ```
3. **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
4. **Configure API keys:**
  - Create a `.env` file in the project root:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
    ```

---

## 🚦 Usage

1. **Start the Streamlit app:**
  ```bash
  streamlit run streamlit_app.py
  ```
2. **Upload PDF documents** via the sidebar.
3. **Configure chunk size and overlap** as needed.
4. **Click "Process Documents"** to build the knowledge base.
5. **Ask questions** in the chat interface!

### Example Questions
- "What is the summary of all documents?"
- "What is the phone number in the contract?"
- "Summarize the main findings."
- "Search for recent AI news."
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
