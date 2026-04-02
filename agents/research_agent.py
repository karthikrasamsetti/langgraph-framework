"""
Research specialist agent.
Handles anything that needs information finding —
web search, document lookup, current events.
"""
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.factory import get_llm
from observability.logger import log, timed_node
from config.settings import get_settings

settings = get_settings()


# ── State ──────────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    task:           str
    result:         Optional[str]
    error:          Optional[str]
    iteration_count: int


# ── Tools for this agent only ──────────────────────────────────────────────────
def _get_research_tools():
    from tools.registry import web_search, search_documents
    tools = [web_search]
    if settings.RAG_ENABLED:
        tools.append(search_documents)
    return tools


# ── LLM with tools ─────────────────────────────────────────────────────────────
def _get_llm():
    return get_llm().bind_tools(_get_research_tools())


# ── Nodes ──────────────────────────────────────────────────────────────────────
@timed_node("research_reasoner")
def research_reasoner(state: ResearchState) -> dict:
    if state.get("iteration_count", 0) >= settings.MAX_ITERATIONS:
        return {"error": "max_iterations", "result": "Research reached max steps."}

    system = SystemMessage(content="""You are a research specialist.
Your only job is to find accurate information.
Use search_documents for questions about internal company policies and documents.
Use web_search for current events, news, and public information.
Always be specific in your search queries.
Cite where the information came from.""")

    messages = [system] + state["messages"]

    try:
        llm    = _get_llm()
        response = llm.invoke(messages)
        log.info("research_llm_response",
                 has_tool_calls=bool(getattr(response, "tool_calls", None)))
        return {
            "messages":        [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    except Exception as e:
        log.error("research_reasoner_error", error=str(e))
        return {"error": str(e)}


@timed_node("research_tool_executor")
def research_tool_executor(state: ResearchState) -> dict:
    from langchain_core.messages import ToolMessage
    last_msg   = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", [])

    from tools.registry import TOOL_MAP
    tool_messages = []

    for tc in tool_calls:
        tool_fn = TOOL_MAP.get(tc["name"])
        if not tool_fn:
            result = f"Tool '{tc['name']}' not found."
        else:
            try:
                result = tool_fn.invoke(tc["args"])
                log.info("research_tool_result",
                         tool=tc["name"],
                         preview=str(result)[:80])
            except Exception as e:
                result = f"Tool error: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


@timed_node("research_formatter")
def research_formatter(state: ResearchState) -> dict:
    from langchain_core.messages import AIMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"result": str(msg.content)}
    return {"result": "No research result found."}


# ── Routing ────────────────────────────────────────────────────────────────────
def should_use_tools(state: ResearchState) -> Literal["research_tool_executor", "research_formatter"]:
    if state.get("error"):
        return "research_formatter"
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "research_tool_executor"
    return "research_formatter"


def after_tools(state: ResearchState) -> Literal["research_reasoner", "research_formatter"]:
    if state.get("error"):
        return "research_formatter"
    return "research_reasoner"


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_research_graph():
    g = StateGraph(ResearchState)

    g.add_node("research_reasoner",      research_reasoner)
    g.add_node("research_tool_executor", research_tool_executor)
    g.add_node("research_formatter",     research_formatter)

    g.add_edge(START,                    "research_reasoner")
    g.add_edge("research_formatter",     END)

    g.add_conditional_edges(
        "research_reasoner",
        should_use_tools,
        {
            "research_tool_executor": "research_tool_executor",
            "research_formatter":     "research_formatter",
        }
    )
    g.add_conditional_edges(
        "research_tool_executor",
        after_tools,
        {
            "research_reasoner":  "research_reasoner",
            "research_formatter": "research_formatter",
        }
    )

    return g.compile()


# Singleton
_research_graph = None
def get_research_graph():
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    return _research_graph