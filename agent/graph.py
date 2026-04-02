"""
This is where all nodes get connected into the LangGraph state machine.
The routing logic lives entirely in pure Python functions — easy to test.
"""
from typing import Literal
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.nodes import (
    input_validator,
    llm_reasoner,
    tool_executor,
    response_formatter,
    error_handler,
)
from agent.checkpointer import get_checkpointer
from observability.logger import log


# ── Conditional edge functions ─────────────────────────────────────────────────

def should_use_tools(
    state: AgentState,
) -> Literal["tool_executor", "response_formatter"]:
    """After LLM responds — did it request tool calls?"""
    if state.get("error"):
        return "response_formatter"
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tool_executor"
    return "response_formatter"


def after_tools(
    state: AgentState,
) -> Literal["llm_reasoner", "error_handler"]:
    """After tools execute — loop back to LLM to reason over results."""
    if state.get("error"):
        return "error_handler"
    return "llm_reasoner"


def after_validation(
    state: AgentState,
) -> Literal["llm_reasoner", "error_handler"]:
    if state.get("error"):
        return "error_handler"
    return "llm_reasoner"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def build_graph(interrupt_before: list[str] = None):
    """
    Builds and compiles the agent graph.

    interrupt_before: list of node names to pause at for human review.
    Example: interrupt_before=["tool_executor"] → human approves before any tool runs.
    """
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("input_validator",   input_validator)
    g.add_node("llm_reasoner",      llm_reasoner)
    g.add_node("tool_executor",     tool_executor)
    g.add_node("response_formatter",response_formatter)
    g.add_node("error_handler",     error_handler)

    # Static edges
    g.add_edge(START,                "input_validator")
    g.add_edge("response_formatter", END)
    g.add_edge("error_handler",      END)

    # Conditional edges (the branching logic)
    g.add_conditional_edges(
        "input_validator",
        after_validation,
        {"llm_reasoner": "llm_reasoner", "error_handler": "error_handler"},
    )
    g.add_conditional_edges(
        "llm_reasoner",
        should_use_tools,
        {"tool_executor": "tool_executor", "response_formatter": "response_formatter"},
    )
    g.add_conditional_edges(
        "tool_executor",
        after_tools,
        {"llm_reasoner": "llm_reasoner", "error_handler": "error_handler"},
    )

    checkpointer = get_checkpointer()
    compile_kwargs = {"checkpointer": checkpointer}
    if interrupt_before:
        compile_kwargs["interrupt_before"] = interrupt_before

    app = g.compile(**compile_kwargs)
    log.info("graph_compiled", nodes=list(g.nodes), checkpointer=type(checkpointer).__name__)
    return app


# ── Module-level singleton ─────────────────────────────────────────────────────
_graph = None

def get_graph(interrupt_before: list[str] = None):
    global _graph
    if _graph is None:
        _graph = build_graph(interrupt_before=interrupt_before)
    return _graph