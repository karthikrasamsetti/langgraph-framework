"""
Code specialist agent.
Handles anything that needs computation, math,
or data fetching from APIs.
"""
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.factory import get_llm
from observability.logger import log, timed_node
from config.settings import get_settings

settings = get_settings()


# ── State ──────────────────────────────────────────────────────────────────────
class CodeState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    task:            str
    result:          Optional[str]
    error:           Optional[str]
    iteration_count: int


# ── Tools for this agent only ──────────────────────────────────────────────────
def _get_code_tools():
    from tools.registry import calculator, http_get, get_current_time
    return [calculator, http_get, get_current_time]


def _get_llm():
    return get_llm().bind_tools(_get_code_tools())


# ── Nodes ──────────────────────────────────────────────────────────────────────
@timed_node("code_reasoner")
def code_reasoner(state: CodeState) -> dict:
    if state.get("iteration_count", 0) >= settings.MAX_ITERATIONS:
        return {"error": "max_iterations", "result": "Code agent reached max steps."}

    system = SystemMessage(content="""You are a computation specialist.
Your job is to perform calculations, process data, and fetch information from APIs.
Use calculator for any mathematical expressions — always show the expression and result.
Use http_get to fetch data from REST APIs when a URL is provided.
Use get_current_time when date or time is needed.
Always show your working clearly.""")

    messages = [system] + state["messages"]

    try:
        llm      = _get_llm()
        response = llm.invoke(messages)
        log.info("code_llm_response",
                 has_tool_calls=bool(getattr(response, "tool_calls", None)))
        return {
            "messages":        [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    except Exception as e:
        log.error("code_reasoner_error", error=str(e))
        return {"error": str(e)}


@timed_node("code_tool_executor")
def code_tool_executor(state: CodeState) -> dict:
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
                log.info("code_tool_result",
                         tool=tc["name"],
                         preview=str(result)[:80])
            except Exception as e:
                result = f"Tool error: {e}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages}


@timed_node("code_formatter")
def code_formatter(state: CodeState) -> dict:
    from langchain_core.messages import AIMessage
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"result": str(msg.content)}
    return {"result": "No computation result found."}


# ── Routing ────────────────────────────────────────────────────────────────────
def should_use_tools(state: CodeState) -> Literal["code_tool_executor", "code_formatter"]:
    if state.get("error"):
        return "code_formatter"
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "code_tool_executor"
    return "code_formatter"


def after_tools(state: CodeState) -> Literal["code_reasoner", "code_formatter"]:
    if state.get("error"):
        return "code_formatter"
    return "code_reasoner"


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_code_graph():
    g = StateGraph(CodeState)

    g.add_node("code_reasoner",      code_reasoner)
    g.add_node("code_tool_executor", code_tool_executor)
    g.add_node("code_formatter",     code_formatter)

    g.add_edge(START,                "code_reasoner")
    g.add_edge("code_formatter",     END)

    g.add_conditional_edges(
        "code_reasoner",
        should_use_tools,
        {
            "code_tool_executor": "code_tool_executor",
            "code_formatter":     "code_formatter",
        }
    )
    g.add_conditional_edges(
        "code_tool_executor",
        after_tools,
        {
            "code_reasoner":  "code_reasoner",
            "code_formatter": "code_formatter",
        }
    )

    return g.compile()


_code_graph = None
def get_code_graph():
    global _code_graph
    if _code_graph is None:
        _code_graph = build_code_graph()
    return _code_graph