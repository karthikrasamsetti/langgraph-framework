"""
Supervisor agent — orchestrates specialist agents.
Reads the user message, picks the best specialist,
runs it, and returns the final answer.
"""
import json
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm.factory import get_llm
from observability.logger import log, timed_node
from config.settings import get_settings

settings = get_settings()

# ── Supervisor state ───────────────────────────────────────────────────────────
class SupervisorState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    user_input:      str
    next_agent:      Optional[str]   # research | code | general
    agent_result:    Optional[str]   # what the specialist returned
    final_response:  Optional[str]
    error:           Optional[str]
    iteration_count: int
    metadata:        dict


# ── Node 1: classify intent ────────────────────────────────────────────────────
@timed_node("supervisor_classifier")
def supervisor_classifier(state: SupervisorState) -> dict:
    """
    Uses LLM to classify which specialist should handle this query.
    This is a cheap, fast classification call — not a full reasoning call.
    Returns one of: research | code | general
    """
    system = SystemMessage(content="""You are a task router for an AI assistant system.
Your ONLY job is to classify which specialist agent should handle the user's request.

Agents available:
- research : user needs information, facts, document lookup, web search,
             company policies, current events, news
- code     : user needs math calculation, API calls, data processing,
             time/date queries, HTTP requests
- general  : simple conversation, greetings, explanations from knowledge,
             anything that doesn't need search or calculation

Respond with ONLY a JSON object, nothing else:
{"agent": "research"} or {"agent": "code"} or {"agent": "general"}

Examples:
"What is the refund policy?" → {"agent": "research"}
"Calculate 15% of 200"       → {"agent": "code"}
"What is machine learning?"  → {"agent": "general"}
"Latest IPL scores"          → {"agent": "research"}
"What time is it?"           → {"agent": "code"}
"Hello, how are you?"        → {"agent": "general"}
""")

    user = HumanMessage(content=state["user_input"])

    try:
        llm      = get_llm()   # no tools — just classification
        response = llm.invoke([system, user])
        content  = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        data       = json.loads(content)
        next_agent = data.get("agent", "general")

        # Validate — default to general if unexpected value
        if next_agent not in ("research", "code", "general"):
            next_agent = "general"

        log.info("supervisor_classification",
                 user_input=state["user_input"][:60],
                 routed_to=next_agent)

        return {"next_agent": next_agent}

    except Exception as e:
        log.warning("supervisor_classification_failed",
                    error=str(e),
                    fallback="general")
        return {"next_agent": "general"}


# ── Node 2: run the specialist ─────────────────────────────────────────────────
@timed_node("supervisor_run_agent")
def supervisor_run_agent(state: SupervisorState) -> dict:
    """
    Calls the appropriate specialist graph and captures its result.
    """
    agent     = state.get("next_agent", "general")
    task      = state["user_input"]
    messages  = [HumanMessage(content=task)]

    log.info("supervisor_dispatching", agent=agent, task=task[:60])

    try:
        if agent == "research":
            from agents.research_agent import get_research_graph
            graph        = get_research_graph()
            initial_state = {
                "messages":        messages,
                "task":            task,
                "result":          None,
                "error":           None,
                "iteration_count": 0,
            }
            final = graph.invoke(initial_state)
            result = final.get("result", "Research agent returned no result.")

        elif agent == "code":
            from agents.code_agent import get_code_graph
            graph         = get_code_graph()
            initial_state = {
                "messages":        messages,
                "task":            task,
                "result":          None,
                "error":           None,
                "iteration_count": 0,
            }
            final  = graph.invoke(initial_state)
            result = final.get("result", "Code agent returned no result.")

        else:
            from agents.general_agent import get_general_graph
            graph         = get_general_graph()
            initial_state = {
                "messages":        messages,
                "task":            task,
                "result":          None,
                "error":           None,
                "iteration_count": 0,
            }
            final  = graph.invoke(initial_state)
            result = final.get("result", "General agent returned no result.")

        log.info("supervisor_agent_complete",
                 agent=agent,
                 result_preview=str(result)[:80])
        return {"agent_result": result}

    except Exception as e:
        log.error("supervisor_run_agent_error", agent=agent, error=str(e))
        return {
            "agent_result": None,
            "error":        f"Agent '{agent}' failed: {e}",
        }


# ── Node 3: format final response ──────────────────────────────────────────────
@timed_node("supervisor_formatter")
def supervisor_formatter(state: SupervisorState) -> dict:
    """
    Takes the specialist result and formats it as the final response.
    """
    if state.get("error"):
        return {
            "final_response": f"I encountered an issue: {state['error']}. Please try again.",
        }

    result = state.get("agent_result", "")
    if not result:
        return {"final_response": "I was unable to generate a response. Please try again."}

    return {"final_response": result}


# ── Routing ────────────────────────────────────────────────────────────────────
def after_classification(
    state: SupervisorState,
) -> Literal["supervisor_run_agent", "supervisor_formatter"]:
    if state.get("error"):
        return "supervisor_formatter"
    return "supervisor_run_agent"


def after_run_agent(
    state: SupervisorState,
) -> Literal["supervisor_formatter"]:
    return "supervisor_formatter"


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_supervisor_graph():
    g = StateGraph(SupervisorState)

    g.add_node("supervisor_classifier",  supervisor_classifier)
    g.add_node("supervisor_run_agent",   supervisor_run_agent)
    g.add_node("supervisor_formatter",   supervisor_formatter)

    g.add_edge(START,                    "supervisor_classifier")
    g.add_edge("supervisor_formatter",   END)

    g.add_conditional_edges(
        "supervisor_classifier",
        after_classification,
        {
            "supervisor_run_agent":  "supervisor_run_agent",
            "supervisor_formatter":  "supervisor_formatter",
        }
    )
    g.add_conditional_edges(
        "supervisor_run_agent",
        after_run_agent,
        {"supervisor_formatter": "supervisor_formatter"}
    )

    from agent.checkpointer import get_checkpointer
    return g.compile(checkpointer=get_checkpointer())


_supervisor_graph = None
def get_supervisor_graph():
    global _supervisor_graph
    if _supervisor_graph is None:
        _supervisor_graph = build_supervisor_graph()
    return _supervisor_graph