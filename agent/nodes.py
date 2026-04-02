import json
import time
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool

from agent.state import AgentState
from llm.factory import get_llm
from tools.registry import ALL_TOOLS, TOOL_MAP
from observability.logger import log, timed_node
from config.settings import get_settings

settings = get_settings()

# Bind tools to the LLM once at module load — works identically for all providers
_llm_with_tools = get_llm().bind_tools(ALL_TOOLS)


@timed_node("input_validator")
def input_validator(state: AgentState) -> dict:
    """Sanitizes and normalizes incoming user input."""
    user_input = state["user_input"].strip()
    if not user_input:
        return {"error": "Empty input", "final_response": "Please provide a message."}
    if len(user_input) > 10_000:
        user_input = user_input[:10_000]
        log.warning("input_truncated", original_length=len(state["user_input"]))
    return {
        "user_input":      user_input,
        "messages":        [HumanMessage(content=user_input)],
        "iteration_count": 0,
        "retry_count":     0,
        "tool_results":    [],
        "error":           None,
    }


# @timed_node("llm_reasoner")
# def llm_reasoner(state: AgentState) -> dict:
#     """Core LLM reasoning node — calls the LLM with current message history."""
#     iteration = state.get("iteration_count", 0)
#     if iteration >= state.get("max_iterations", settings.MAX_ITERATIONS):
#         log.warning("max_iterations_reached", iteration=iteration)
#         return {
#             "final_response": "I've reached the maximum number of reasoning steps. "
#                               "Here's my best answer based on what I have so far.",
#             "error": "max_iterations",
#         }

#     system = SystemMessage(content="""You are a helpful, accurate AI assistant.
# You have access to tools. Use them when you need real-time data or computation.
# Think step by step. Be concise.""")

#     messages = [system] + state["messages"]

#     for attempt in range(settings.MAX_RETRIES):
#         try:
#             response = _llm_with_tools.invoke(messages)
#             log.info(
#                 "llm_response",
#                 has_tool_calls=bool(getattr(response, "tool_calls", None)),
#                 iteration=iteration + 1,
#             )
#             return {
#                 "messages":        [response],
#                 "iteration_count": iteration + 1,
#                 "retry_count":     0,
#                 "error":           None,
#             }
#         except Exception as exc:
#             log.warning("llm_retry", attempt=attempt + 1, error=str(exc))
#             if attempt < settings.MAX_RETRIES - 1:
#                 time.sleep(settings.RETRY_DELAY * (attempt + 1))  # exponential backoff
#             else:
#                 return {"error": f"LLM failed after {settings.MAX_RETRIES} attempts: {exc}"}


@timed_node("llm_reasoner")
def llm_reasoner(state: AgentState) -> dict:
    iteration = state.get("iteration_count", 0)
    if iteration >= state.get("max_iterations", settings.MAX_ITERATIONS):
        return {
            "final_response": "Reached maximum reasoning steps.",
            "error": "max_iterations",
        }

    # ── Build system prompt ────────────────────────────────────────────────────
    # Check if documents are ingested so we can tell the LLM about them
    doc_context = ""
    if settings.RAG_ENABLED:
        try:
            from rag.retriever import get_collection_stats
            stats = get_collection_stats()
            chunk_count = stats.get("total_chunks", 0)
            if chunk_count > 0:
                doc_context = f"""
You have access to a document knowledge base containing {chunk_count} indexed chunks.
ALWAYS use the search_documents tool when the user asks about:
- Policies (refund, leave, expense, IT, HR, working hours)
- Any topic that could be in an internal document
- Questions about company rules, procedures or guidelines
- Any question where document knowledge would be helpful

Do NOT ask the user to provide documents — they are already indexed and searchable.
"""
        except Exception:
            pass

    system_content = f"""You are a helpful AI assistant with access to the following tools:

1. search_documents — searches the internal document knowledge base
2. web_search — searches the internet for current information  
3. calculator — evaluates mathematical expressions
4. get_current_time — returns the current date and time
5. http_get — makes HTTP GET requests
{doc_context}
Always use the most appropriate tool. Think step by step. Be concise."""

    system   = SystemMessage(content=system_content)
    messages = [system] + state["messages"]

    for attempt in range(settings.MAX_RETRIES):
        try:
            response = _llm_with_tools.invoke(messages)
            log.info(
                "llm_response",
                has_tool_calls=bool(getattr(response, "tool_calls", None)),
                iteration=iteration + 1,
            )
            return {
                "messages":        [response],
                "iteration_count": iteration + 1,
                "retry_count":     0,
                "error":           None,
            }
        except Exception as exc:
            log.warning("llm_retry", attempt=attempt + 1, error=str(exc))
            if attempt < settings.MAX_RETRIES - 1:
                time.sleep(settings.RETRY_DELAY * (attempt + 1))
            else:
                return {
                    "error": f"LLM failed after {settings.MAX_RETRIES} attempts: {exc}"
                }

@timed_node("tool_executor")
def tool_executor(state: AgentState) -> dict:
    """Executes all tool calls requested by the LLM in the last message."""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return {}

    tool_messages = []
    tool_results  = list(state.get("tool_results", []))

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_fn: BaseTool = TOOL_MAP.get(tool_name)

        if not tool_fn:
            result = f"Tool '{tool_name}' not found."
            log.warning("tool_not_found", tool=tool_name)
        else:
            try:
                log.info("tool_call", tool=tool_name, args=tool_args)
                result = tool_fn.invoke(tool_args)
                tool_results.append({"tool": tool_name, "args": tool_args, "result": result})
                log.info("tool_result", tool=tool_name, result_preview=str(result)[:100])
            except Exception as exc:
                result = f"Tool '{tool_name}' error: {exc}"
                log.error("tool_error", tool=tool_name, error=str(exc))

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"])
        )

    return {"messages": tool_messages, "tool_results": tool_results}


@timed_node("response_formatter")
def response_formatter(state: AgentState) -> dict:
    """Extracts the final text response from the last AI message."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return {"final_response": str(msg.content)}
    return {"final_response": "No response generated."}


@timed_node("error_handler")
def error_handler(state: AgentState) -> dict:
    """Gracefully handles errors — logs and returns a user-friendly message."""
    error = state.get("error", "Unknown error")
    log.error("agent_error", error=error, iteration=state.get("iteration_count"))
    return {
        "final_response": f"I encountered an issue: {error}. Please try again.",
    }