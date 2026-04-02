"""
General conversation agent.
Handles simple questions, explanations, greetings.
No tools — answers purely from LLM knowledge.
"""
from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.factory import get_llm
from observability.logger import log, timed_node
from config.settings import get_settings

settings = get_settings()


# ── State ──────────────────────────────────────────────────────────────────────
class GeneralState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    task:            str
    result:          Optional[str]
    error:           Optional[str]
    iteration_count: int


# ── Nodes ──────────────────────────────────────────────────────────────────────
@timed_node("general_reasoner")
def general_reasoner(state: GeneralState) -> dict:
    system = SystemMessage(content="""You are a helpful conversational assistant.
Answer questions clearly and concisely from your knowledge.
You do not have access to search tools or calculators.
For anything requiring real-time data or calculations,
let the user know they should ask specifically for that.""")

    messages = [system] + state["messages"]

    try:
        llm      = get_llm()   # no tools bound
        response = llm.invoke(messages)
        log.info("general_llm_response")
        return {
            "messages":        [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    except Exception as e:
        log.error("general_reasoner_error", error=str(e))
        return {"error": str(e)}


@timed_node("general_formatter")
def general_formatter(state: GeneralState) -> dict:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"result": str(msg.content)}
    return {"result": "I could not generate a response."}


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_general_graph():
    g = StateGraph(GeneralState)

    g.add_node("general_reasoner",  general_reasoner)
    g.add_node("general_formatter", general_formatter)

    g.add_edge(START,               "general_reasoner")
    g.add_edge("general_reasoner",  "general_formatter")
    g.add_edge("general_formatter", END)

    return g.compile()


_general_graph = None
def get_general_graph():
    global _general_graph
    if _general_graph is None:
        _general_graph = build_general_graph()
    return _general_graph