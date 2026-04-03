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

from api.auth import require_api_key
from fastapi import Depends

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

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     log.info("startup",
#              provider=settings.LLM_PROVIDER,
#              checkpointer=settings.CHECKPOINTER,
#              langsmith=LANGSMITH_ENABLED)

#     # ── Pre-warm everything at startup ────────────────────────────
#     get_graph()

#     if settings.RAG_ENABLED:
#         try:
#             from rag.embeddings import get_embedding_function
#             from rag.retriever import get_collection, get_collection_stats
#             print("[RAG] Pre-loading embedding model...")
#             get_embedding_function()          # loads model into RAM now
#             get_collection()                  # opens ChromaDB connection now
#             stats = get_collection_stats()
#             print(f"[RAG] Ready — {stats.get('total_chunks', 0)} chunks indexed")
#             log.info("rag_ready",
#                      chunks=stats.get("total_chunks", 0),
#                      collection=stats.get("collection"))
#         except Exception as e:
#             log.warning("rag_init_failed", error=str(e))

#     yield
#     log.info("shutdown")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup",
             provider=settings.LLM_PROVIDER,
             checkpointer=settings.CHECKPOINTER,
             langsmith=LANGSMITH_ENABLED)

    get_graph()   # warm up single agent

    # ── Warm up multi-agent system ─────────────────────────────────
    try:
        from agents.supervisor import get_supervisor_graph
        from agents.research_agent import get_research_graph
        from agents.code_agent import get_code_graph
        from agents.general_agent import get_general_graph
        get_supervisor_graph()
        get_research_graph()
        get_code_graph()
        get_general_graph()
        log.info("multi_agent_ready")
    except Exception as e:
        log.warning("multi_agent_init_failed", error=str(e))

    if settings.RAG_ENABLED:
        try:
            from rag.embeddings import get_embedding_function
            from rag.retriever import get_collection, get_collection_stats
            print("[RAG] Pre-loading embedding model...")
            get_embedding_function()
            get_collection()
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

# ── Rate limit header middleware ───────────────────────────────────────────────
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class RateLimitHeaderMiddleware(BaseHTTPMiddleware):
    """
    Attaches X-RateLimit-* headers to every response
    when they were set by the require_api_key dependency.
    """
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        headers  = getattr(request.state, "rate_limit_headers", {})
        for key, val in headers.items():
            response.headers[key] = val
        return response

app.add_middleware(RateLimitHeaderMiddleware)

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

# ── Human-in-the-loop models ───────────────────────────────────────────────────
class PendingToolCall(BaseModel):
    name: str
    args: dict


class PendingApproval(BaseModel):
    session_id:   str
    mode:         str          # "single" or "multi"
    pending_node: str          # which node is paused
    next_agent:   Optional[str] = None      # for multi-agent
    pending_tools: list[PendingToolCall] = []  # for single agent
    message:      str          # human-readable description


class ResumeRequest(BaseModel):
    action:     str            # "approve" | "reject" | "override"
    next_agent: Optional[str] = None   # for override in multi-agent
    mode:       str = "single"         # "single" or "multi"


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
async def chat(req: ChatRequest, key_meta: dict = Depends(require_api_key),):
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
        final_state = await asyncio.to_thread(
            graph.invoke, initial_state, config
        )

        # ── Check if graph paused at an interrupt point ────────────────────────
        current_state = graph.get_state(config)
        if current_state.next:
            pending_node = current_state.next[0]
            log.info("graph_paused",
                     session_id=session_id,
                     pending_node=pending_node)

            # Return pending status — UI will poll /chat/pending
            return ChatResponse(
                session_id=session_id,
                response="",
                tool_results=[],
                iteration_count=final_state.get("iteration_count", 0),
                error=f"pending:{pending_node}",
            )

        trace_url = (
            f"https://smith.langchain.com/projects/{settings.LANGSMITH_PROJECT}"
            if LANGSMITH_ENABLED else None
        )

        log.info("chat_complete",
                 session_id=session_id,
                 iterations=final_state.get("iteration_count", 0),
                 tools=len(final_state.get("tool_results", [])))

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
async def chat_stream(req: ChatRequest, key_meta: dict = Depends(require_api_key),):
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
async def submit_feedback(req: FeedbackRequest,key_meta: dict = Depends(require_api_key), ):
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
async def get_session_history(session_id: str, key_meta: dict = Depends(require_api_key),):
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
    

# ── POST /multi-agent/chat ────────────────────────────────────────────────────
class MultiAgentResponse(BaseModel):
    session_id:      str
    response:        str
    agent_used:      Optional[str] = None
    iteration_count: int           = 0
    error:           Optional[str] = None
    trace_url:       Optional[str] = None


@app.post("/multi-agent/chat", response_model=MultiAgentResponse)
async def multi_agent_chat(req: ChatRequest,key_meta: dict = Depends(require_api_key), ):
    """
    Multi-agent endpoint — supervisor routes to the best specialist.
    research agent  → web search + document search
    code agent      → calculator + http + time
    general agent   → conversation from knowledge
    """
    from agents.supervisor import get_supervisor_graph, SupervisorState

    session_id = req.session_id or str(uuid.uuid4())
    graph      = get_supervisor_graph()

    initial_state = SupervisorState(
        messages=[],
        user_input=req.message,
        next_agent=None,
        agent_result=None,
        final_response=None,
        error=None,
        iteration_count=0,
        metadata={
            "session_id": session_id,
            "user_id":    req.user_id or "anonymous",
        },
    )

    config = {
        "configurable": {"thread_id": f"multi-{session_id}"},
        "run_name":     f"multi-agent · {session_id[:8]}",
        "tags":         [settings.LLM_PROVIDER, "multi-agent"],
        "metadata": {
            "session_id": session_id,
            "provider":   settings.LLM_PROVIDER,
            "mode":       "multi-agent",
        },
    }

    try:
        final_state = await asyncio.to_thread(
            graph.invoke, initial_state, config
        )

        # ── Check if graph paused at interrupt point ───────────────────────────
        current_state = graph.get_state(config)
        if current_state.next:
            pending_node = current_state.next[0]
            log.info("supervisor_paused",
                     session_id=session_id,
                     pending_node=pending_node)
            return MultiAgentResponse(
                session_id=session_id,
                response="",
                agent_used=final_state.get("next_agent"),
                iteration_count=0,
                error=f"pending:{pending_node}",
            )

        # ── Graph completed normally ───────────────────────────────────────────
        trace_url = (
            f"https://smith.langchain.com/projects/{settings.LANGSMITH_PROJECT}"
            if LANGSMITH_ENABLED else None
        )

        log.info("multi_agent_complete",
                 session_id=session_id,
                 agent_used=final_state.get("next_agent"),
                 trace_url=trace_url)

        return MultiAgentResponse(
            session_id=session_id,
            response=final_state.get("final_response", ""),
            agent_used=final_state.get("next_agent"),
            iteration_count=final_state.get("iteration_count", 0),
            error=final_state.get("error"),
            trace_url=trace_url,
        )

    except Exception as exc:
        log.error("multi_agent_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))
    
    
# ── GET /chat/pending/{session_id} ────────────────────────────────────────────
@app.get("/chat/pending/{session_id}")
async def get_pending(session_id: str, mode: str = "single",key_meta: dict = Depends(require_api_key), ):
    """
    Check if a session is paused waiting for human approval.
    Returns the pending node and what it's about to do.
    """
    try:
        if mode == "multi":
            from agents.supervisor import get_supervisor_graph
            graph  = get_supervisor_graph()
            config = {"configurable": {"thread_id": f"multi-{session_id}"}}
        else:
            graph  = get_graph()
            config = {"configurable": {"thread_id": session_id}}

        state = graph.get_state(config)

        # state.next is empty tuple when graph finished
        # non-empty means graph is paused at that node
        if not state.next:
            return {"status": "completed", "session_id": session_id}

        pending_node = state.next[0]
        values       = state.values

        # ── Single agent — extract pending tool calls ──────────────────────────
        if mode == "single" and pending_node == "tool_executor":
            last_msg   = values.get("messages", [])[-1] if values.get("messages") else None
            tool_calls = getattr(last_msg, "tool_calls", []) if last_msg else []

            pending_tools = [
                PendingToolCall(name=tc["name"], args=tc["args"])
                for tc in tool_calls
            ]

            tools_desc = ", ".join(
                f"{t.name}({list(t.args.values())[0] if t.args else ''})"
                for t in pending_tools
            )

            return PendingApproval(
                session_id=session_id,
                mode="single",
                pending_node=pending_node,
                pending_tools=pending_tools,
                message=f"Agent wants to call: {tools_desc}",
            )

        # ── Multi-agent — extract routing decision ─────────────────────────────
        elif mode == "multi" and pending_node == "supervisor_run_agent":
            next_agent = values.get("next_agent", "unknown")
            return PendingApproval(
                session_id=session_id,
                mode="multi",
                pending_node=pending_node,
                next_agent=next_agent,
                message=f"Supervisor chose: {next_agent} agent",
            )

        else:
            return PendingApproval(
                session_id=session_id,
                mode=mode,
                pending_node=pending_node,
                message=f"Graph paused before: {pending_node}",
            )

    except Exception as e:
        log.error("get_pending_error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=404, detail=str(e))


# ── POST /chat/resume/{session_id} ────────────────────────────────────────────
@app.post("/chat/resume/{session_id}", response_model=ChatResponse)
async def resume_chat(session_id: str, req: ResumeRequest,key_meta: dict = Depends(require_api_key), ):
    """
    Resume a paused graph.

    action=approve  → continue as planned
    action=reject   → stop, return rejection message
    action=override → change routing then continue (multi-agent only)
    """
    try:
        if req.mode == "multi":
            from agents.supervisor import get_supervisor_graph, SupervisorState
            graph  = get_supervisor_graph()
            config = {"configurable": {"thread_id": f"multi-{session_id}"}}
        else:
            graph  = get_graph()
            config = {"configurable": {"thread_id": session_id}}

        # ── Reject ────────────────────────────────────────────────────────────
        if req.action == "reject":
            log.info("human_rejected",
                     session_id=session_id,
                     mode=req.mode)
            return ChatResponse(
                session_id=session_id,
                response="Action rejected by human reviewer. The agent has been stopped.",
                tool_results=[],
                iteration_count=0,
                error="rejected_by_human",
            )

        # ── Override (multi-agent only) ───────────────────────────────────────
        if req.action == "override" and req.next_agent:
            if req.next_agent not in ("research", "code", "general"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid agent: {req.next_agent}. Choose research, code, or general."
                )
            # Update state with the human's routing choice
            graph.update_state(
                config,
                {"next_agent": req.next_agent},
            )
            log.info("human_override",
                     session_id=session_id,
                     new_agent=req.next_agent)

        # ── Approve or override — resume graph ────────────────────────────────
        log.info("human_approved",
                 session_id=session_id,
                 action=req.action,
                 mode=req.mode)

        final_state = await asyncio.to_thread(
            graph.invoke, None, config   # None = no new input, just resume
        )

        # Check if graph paused again (another interrupt point)
        current_state = graph.get_state(config)
        if current_state.next:
            # Graph paused at another node — return pending status
            return ChatResponse(
                session_id=session_id,
                response="",
                tool_results=[],
                iteration_count=final_state.get("iteration_count", 0),
                error=f"pending:{current_state.next[0]}",
            )

        # Graph completed
        if req.mode == "multi":
            response_text = final_state.get("final_response", "")
        else:
            response_text = final_state.get("final_response", "")

        trace_url = (
            f"https://smith.langchain.com/projects/{settings.LANGSMITH_PROJECT}"
            if LANGSMITH_ENABLED else None
        )

        log.info("resume_complete",
                 session_id=session_id,
                 mode=req.mode)

        return ChatResponse(
            session_id=session_id,
            response=response_text,
            tool_results=final_state.get("tool_results", []),
            iteration_count=final_state.get("iteration_count", 0),
            error=None,
            trace_url=trace_url,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("resume_error", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    
# ── GET /api/auth/validate ────────────────────────────────────────────────────
@app.get("/api/auth/validate")
async def validate_key(key_meta: dict = Depends(require_api_key)):
    """
    Test endpoint — validates an API key and returns its metadata.
    Use this to confirm your key works before making real requests.
    """
    return {
        "valid":    True,
        "name":     key_meta["name"],
        "rpm":      key_meta["rpm"],
        "message":  f"Key is valid. Rate limit: {key_meta['rpm']} req/min.",
    }