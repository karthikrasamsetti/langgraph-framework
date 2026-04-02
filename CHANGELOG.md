# Changelog

## v1.0.0 — Initial release

### Added
- LangGraph agent framework with StateGraph
- Multi-provider LLM factory — OpenAI, Anthropic, Bedrock, Groq, Ollama, HuggingFace, Azure
- Tool registry — web_search (Tavily), calculator, get_current_time, http_get, search_documents
- RAG pipeline — ChromaDB vector store, local and OpenAI embeddings
- Document ingestion — PDF, TXT, DOCX, MD, URL support
- FastAPI server — /chat, /chat/stream, /health, /feedback, /sessions endpoints
- Streaming SSE support
- LangSmith tracing integration
- Structured logging with structlog
- MemorySaver, SqliteSaver, PostgresSaver checkpointing
- HTML chat UI — zero build step
- Streamlit dashboard with streaming toggle
- Unit tests for all nodes and routing logic
- Windows ANSI color fix
- CORS middleware