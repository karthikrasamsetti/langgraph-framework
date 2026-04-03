"""
Graph assembly — wires all nodes together.
interrupt_before is now configurable via settings.
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
from config.settings import get_settings, get_human_in_loop_nodes

settings = get_settings()


# ── Conditional edge functions ─────────────────────────────────────────────────
def should_use_tools(
    state: AgentState,
) -> Literal["tool_executor", "response_formatter"]:
    if state.get("error"):
        return "response_formatter"
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tool_executor"
    return "response_formatter"


def after_tools(
    state: AgentState,
) -> Literal["llm_reasoner", "error_handler"]:
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
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("input_validator",    input_validator)
    g.add_node("llm_reasoner",       llm_reasoner)
    g.add_node("tool_executor",      tool_executor)
    g.add_node("response_formatter", response_formatter)
    g.add_node("error_handler",      error_handler)

    g.add_edge(START,                "input_validator")
    g.add_edge("response_formatter", END)
    g.add_edge("error_handler",      END)

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

    # ── interrupt_before ───────────────────────────────────────────────────────
    # Only add nodes that actually exist in THIS graph
    # supervisor_run_agent is in supervisor graph not here
    all_interrupt_nodes = get_human_in_loop_nodes()
    valid_single_agent_nodes = {
        "input_validator", "llm_reasoner",
        "tool_executor", "response_formatter", "error_handler"
    }
    interrupt_nodes = [
        n for n in all_interrupt_nodes
        if n in valid_single_agent_nodes
    ]

    compile_kwargs = {"checkpointer": checkpointer}
    if interrupt_nodes:
        compile_kwargs["interrupt_before"] = interrupt_nodes
        log.info("human_in_loop_enabled",
                 graph="single_agent",
                 interrupt_before=interrupt_nodes)

    app = g.compile(**compile_kwargs)
    log.info("graph_compiled",
             nodes=list(g.nodes),
             checkpointer=type(checkpointer).__name__)
    return app


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph