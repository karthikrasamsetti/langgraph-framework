# LangGraph Production Framework

A production-ready AI agent framework built with LangGraph. Supports multiple LLM providers, RAG with vector search, LangSmith tracing, and a full chat UI.

## Features

- Multi-provider LLM support — OpenAI, Anthropic, AWS Bedrock, Groq, Ollama, HuggingFace, Azure
- RAG (Retrieval Augmented Generation) — PDF, TXT, DOCX, URL ingestion
- LangSmith tracing — full observability for every run
- FastAPI backend — REST + streaming endpoints
- Chat UI — HTML + Streamlit interfaces
- Persistent sessions — memory, SQLite, or PostgreSQL
- Structured logging — pretty dev output, JSON for production

## Project Structure
```
langgraph_framework/
├── api/
│   └── server.py          # FastAPI app — all HTTP endpoints
├── agent/
│   ├── state.py           # AgentState TypedDict — shared state schema
│   ├── nodes.py           # All graph node functions
│   ├── graph.py           # Graph assembly and compilation
│   └── checkpointer.py    # Persistence backend selector
├── config/
│   └── settings.py        # All config via Pydantic BaseSettings
├── llm/
│   └── factory.py         # Universal LLM adapter — swap provider via .env
├── rag/
│   ├── embeddings.py      # Embedding model factory (local / OpenAI)
│   ├── retriever.py       # ChromaDB search logic
│   └── ingest.py          # CLI — load documents into vector DB
├── tools/
│   └── registry.py        # All agent tools — web_search, calculator etc.
├── observability/
│   └── logger.py          # Structured logging setup
├── ui/
│   ├── index.html         # Standalone chat UI (no build step)
│   └── streamlit_app.py   # Streamlit dashboard
├── tests/
│   └── test_agent.py      # Unit tests
├── my_docs/               # Put your documents here (gitignored)
├── .env.example           # Copy to .env and fill in your keys
└── requirements.txt
```

## Quick Start

### 1. Clone and install
```bash
git clone https://github.com/your-username/langgraph-framework.git
cd langgraph-framework
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env — set your LLM_PROVIDER and API keys
```

### 3. Start the server
```bash
uvicorn api.server:app --reload --port 8000
```

### 4. Open the UI

Open `ui/index.html` in your browser or run Streamlit:
```bash
pip install streamlit
streamlit run ui/streamlit_app.py
```

## RAG — Document Search

### Ingest documents
```bash
# Ingest a folder
python rag/ingest.py --path ./my_docs/

# Ingest a single file
python rag/ingest.py --path ./my_docs/report.pdf

# Ingest a webpage
python rag/ingest.py --url https://example.com/docs

# Check what is indexed
python rag/ingest.py --stats

# Clear everything
python rag/ingest.py --clear
```

### Supported file types

| Type | Extension |
|------|-----------|
| PDF | .pdf |
| Text | .txt |
| Markdown | .md |
| Word | .docx |
| Web | URL |

### Switch to OpenAI embeddings (production)
```bash
# In .env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-xxxx
```

Re-run ingest after switching — embeddings from different models are not compatible.

## API Reference

### `GET /health`

Check server status.

**Response**
```json
{
  "status": "ok",
  "provider": "bedrock",
  "langsmith_enabled": true,
  "langsmith_project": "langgraph-production"
}
```

### `POST /chat`

Send a message to the agent.

**Request**
```json
{
  "message": "What is the refund policy?",
  "session_id": "optional-existing-session-id",
  "user_id": "optional-user-id"
}
```

**Response**
```json
{
  "session_id": "A3F9BC12",
  "response": "Based on the policy document...",
  "tool_results": [
    {
      "tool": "search_documents",
      "args": {"query": "refund policy"},
      "result": "..."
    }
  ],
  "iteration_count": 2,
  "error": null,
  "trace_url": "https://smith.langchain.com/projects/langgraph-production"
}
```

### `POST /chat/stream`

Stream the response as server-sent events.

**Request** — same as `/chat`

**Response** — SSE stream
```
data: {"session_id": "A3F9BC12"}
data: {"response": "Based on the policy..."}
data: [DONE]
```

### `GET /sessions/{session_id}/history`

Get full message history for a session.

**Response**
```json
{
  "session_id": "A3F9BC12",
  "messages": [
    {"role": "HumanMessage", "content": "What is the refund policy?"},
    {"role": "AIMessage", "content": "Based on the document..."}
  ]
}
```

### `POST /feedback`

Send a score for a run to LangSmith.

**Request**
```json
{
  "run_id": "uuid-of-run",
  "score": 1.0,
  "comment": "Perfect answer"
}
```

## LLM Providers

Switch provider by changing `LLM_PROVIDER` in `.env` — no code changes.

| Provider | LLM_PROVIDER value | Notes |
|---|---|---|
| OpenAI | `openai` | GPT-4o, GPT-4o-mini |
| Anthropic | `anthropic` | Claude 3.5 Sonnet |
| AWS Bedrock | `bedrock` | Claude, Titan, Llama via AWS |
| Groq | `groq` | Fast inference, use mixtral for tools |
| Ollama | `ollama` | Local models, no API key needed |
| HuggingFace | `huggingface` | Any HF model |
| Azure OpenAI | `azure_openai` | OpenAI behind Azure |

## LangSmith Tracing
```bash
# In .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_xxxx
LANGSMITH_PROJECT=langgraph-production
```

Get your API key at [smith.langchain.com](https://smith.langchain.com).

## Running Tests
```bash
pytest tests/ -v
```

## License

MIT