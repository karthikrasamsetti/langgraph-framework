import sys
import os

# ── Windows ANSI fix ──────────────────────────────────────────────────────────
if sys.platform == "win32":
    os.system("color")
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

# ── LangSmith MUST be set before any langchain/langgraph imports ──────────────
from dotenv import load_dotenv
load_dotenv()

def _setup_langsmith() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    if not tracing:
        print("[LangSmith] Tracing DISABLED")
        return False

    api_key  = os.getenv("LANGSMITH_API_KEY", "")
    project  = os.getenv("LANGSMITH_PROJECT", "langgraph-production")
    endpoint = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    if not api_key:
        print("[LangSmith] WARNING: LANGSMITH_API_KEY is empty — tracing disabled")
        return False

    for key, val in {
        "LANGSMITH_TRACING":    "true",
        "LANGSMITH_API_KEY":    api_key,
        "LANGSMITH_PROJECT":    project,
        "LANGSMITH_ENDPOINT":   endpoint,
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_API_KEY":    api_key,
        "LANGCHAIN_PROJECT":    project,
        "LANGCHAIN_ENDPOINT":   endpoint,
    }.items():
        os.environ[key] = val

    print(f"[LangSmith] Tracing ENABLED")
    print(f"[LangSmith] Project  : {project}")
    print(f"[LangSmith] Key      : {api_key[:12]}...{api_key[-4:]}")
    print(f"[LangSmith] Endpoint : {endpoint}")
    return True

LANGSMITH_ENABLED = _setup_langsmith()

# ── Now safe to import everything else ────────────────────────────────────────
import asyncio
import uuid
import json
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.graph import get_graph
from agent.state import AgentState
from config.settings import get_settings
from observability.logger import log

settings = get_settings()

# ── LangSmith client (optional — only used if tracing enabled) ────────────────
_langsmith_client = None
if LANGSMITH_ENABLED:
    try:
        from langsmith import Client as LangSmithClient
        _langsmith_client = LangSmithClient(
            api_key=os.environ["LANGSMITH_API_KEY"],
            api_url=os.environ["LANGSMITH_ENDPOINT"],
        )
        print("[LangSmith] Client initialized OK")
    except Exception as e:
        print(f"[LangSmith] Client init failed: {e} — tracing will still work via env vars")


# ── Lifespan ──────────────────────────────────────────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     log.info("startup",
#              provider=settings.LLM_PROVIDER,
#              checkpointer=settings.CHECKPOINTER,
#              langsmith=LANGSMITH_ENABLED)
#     get_graph()
#     yield
#     log.info("shutdown")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup",
             provider=settings.LLM_PROVIDER,
             checkpointer=settings.CHECKPOINTER,
             langsmith=LANGSMITH_ENABLED)

    # ── Pre-warm everything at startup ────────────────────────────
    get_graph()

    if settings.RAG_ENABLED:
        try:
            from rag.embeddings import get_embedding_function
            from rag.retriever import get_collection, get_collection_stats
            print("[RAG] Pre-loading embedding model...")
            get_embedding_function()          # loads model into RAM now
            get_collection()                  # opens ChromaDB connection now
            stats = get_collection_stats()
            print(f"[RAG] Ready — {stats.get('total_chunks', 0)} chunks indexed")
            log.info("rag_ready",
                     chunks=stats.get("total_chunks", 0),
                     collection=stats.get("collection"))
        except Exception as e:
            log.warning("rag_init_failed", error=str(e))

    yield
    log.info("shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LangGraph Production Framework",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str           = Field(..., min_length=1, max_length=10_000)
    session_id: Optional[str] = None
    user_id:    Optional[str] = None
    stream:     bool          = False


class ChatResponse(BaseModel):
    session_id:      str
    response:        str
    tool_results:    list
    iteration_count: int
    error:           Optional[str]
    trace_url:       Optional[str] = None


class FeedbackRequest(BaseModel):
    run_id:  str
    score:   float
    comment: Optional[str] = None


# ── Helper: build LangGraph config ────────────────────────────────────────────
def _build_config(session_id: str, user_id: Optional[str] = None) -> dict:
    """
    RunnableConfig fields run_name / tags / metadata are forwarded
    by langgraph to LangSmith automatically — no extra context manager needed.
    """
    return {
        "configurable": {
            "thread_id": session_id,
        },
        "run_name": f"chat · {session_id[:8]}",
        "tags":     [settings.LLM_PROVIDER, "api"],
        "metadata": {
            "session_id": session_id,
            "user_id":    user_id or "anonymous",
            "provider":   settings.LLM_PROVIDER,
        },
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":            "ok",
        "provider":          settings.LLM_PROVIDER,
        "langsmith_enabled": LANGSMITH_ENABLED,
        "langsmith_project": settings.LANGSMITH_PROJECT if LANGSMITH_ENABLED else None,
    }


# ── POST /chat ────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    graph      = get_graph()

    initial_state = AgentState(
        messages=[],
        user_input=req.message,
        intent=None,
        confidence=None,
        next_node=None,
        iteration_count=0,
        retry_count=0,
        max_iterations=settings.MAX_ITERATIONS,
        error=None,
        tool_results=[],
        final_response=None,
        metadata={
            "session_id": session_id,
            "user_id":    req.user_id or "anonymous",
        },
    )

    config = _build_config(session_id, req.user_id)

    try:
        # ── Run the graph ──────────────────────────────────────────
        # LangSmith tracing is fully automatic via the env vars we set above.
        # langgraph reads LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY at invoke time
        # and sends traces in a background thread — no context manager needed.
        final_state = await asyncio.to_thread(
            graph.invoke, initial_state, config
        )

        trace_url = (
            f"https://smith.langchain.com/projects/{settings.LANGSMITH_PROJECT}"
            if LANGSMITH_ENABLED else None
        )

        log.info("chat_complete",
                 session_id=session_id,
                 iterations=final_state.get("iteration_count", 0),
                 tools=len(final_state.get("tool_results", [])),
                 trace_url=trace_url)

        return ChatResponse(
            session_id=session_id,
            response=final_state.get("final_response", ""),
            tool_results=final_state.get("tool_results", []),
            iteration_count=final_state.get("iteration_count", 0),
            error=final_state.get("error"),
            trace_url=trace_url,
        )

    except Exception as exc:
        log.error("chat_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /chat/stream ─────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    graph      = get_graph()

    initial_state = AgentState(
        messages=[], user_input=req.message,
        intent=None, confidence=None, next_node=None,
        iteration_count=0, retry_count=0,
        max_iterations=settings.MAX_ITERATIONS,
        error=None, tool_results=[], final_response=None,
        metadata={"session_id": session_id},
    )
    config = _build_config(session_id, req.user_id)

    async def token_stream() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'session_id': session_id})}\n\n"
        try:
            for chunk in graph.stream(initial_state, config, stream_mode="values"):
                final_resp = chunk.get("final_response")
                if final_resp:
                    yield f"data: {json.dumps({'response': final_resp})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ── POST /feedback ────────────────────────────────────────────────────────────
@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Send thumbs up/down score for a run back to LangSmith."""
    if not _langsmith_client:
        return {"status": "langsmith not enabled"}
    try:
        _langsmith_client.create_feedback(
            run_id=req.run_id,
            key="user_score",
            score=req.score,
            comment=req.comment,
        )
        return {"status": "ok"}
    except Exception as e:
        log.error("feedback_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /sessions/{session_id}/history ────────────────────────────────────────
@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    graph  = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        state    = graph.get_state(config)
        messages = [
            {"role": type(m).__name__, "content": m.content}
            for m in state.values.get("messages", [])
        ]
        return {"session_id": session_id, "messages": messages}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))