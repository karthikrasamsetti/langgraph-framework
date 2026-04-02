from typing import TypedDict, Annotated, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    The single source of truth flowing through every graph node.
    add_messages reducer appends to messages list instead of overwriting.
    """
    # Core conversation
    messages:        Annotated[list[BaseMessage], add_messages]
    user_input:      str

    # Routing + control
    intent:          Optional[str]
    confidence:      Optional[float]
    next_node:       Optional[str]

    # Execution control
    iteration_count: int
    retry_count:     int
    max_iterations:  int
    error:           Optional[str]

    # Tool results
    tool_results:    list[dict[str, Any]]

    # Output
    final_response:  Optional[str]
    metadata:        dict[str, Any]   # arbitrary context — session_id, user_id, etc.