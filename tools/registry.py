"""
Add your tools here. They automatically become available to the agent.
Each tool is a plain Python function with a @tool decorator.
"""
import json
import httpx
from datetime import datetime
from langchain_core.tools import tool
from typing import Optional
from config.settings import get_settings   # ← add this import

from rag.retriever import retrieve, get_collection_stats

settings = get_settings()

@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""
    if not settings.TAVILY_API_KEY:
        return "Web search is not configured. Add TAVILY_API_KEY to your .env file."

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        results = client.search(query, max_results=3)
        return json.dumps([r["content"] for r in results.get("results", [])])
    except ImportError:
        return "Tavily package not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Search failed: {e}"


@tool
def get_current_time(timezone: Optional[str] = "UTC") -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression. Example: '2 + 2 * 10'"""
    try:
        # Safe eval — only math ops
        allowed = set("0123456789+-*/().,% ")
        if not all(c in allowed for c in expression):
            return "Error: expression contains unsafe characters"
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def http_get(url: str, headers: Optional[str] = None) -> str:
    """Make an HTTP GET request and return the response body."""
    try:
        h = json.loads(headers) if headers else {}
        response = httpx.get(url, headers=h, timeout=10)
        return response.text[:2000]   # truncate long responses
    except Exception as e:
        return f"HTTP error: {e}"

@tool
def search_documents(query: str) -> str:
    """
    Search through ingested documents to find relevant information.
    Use this when the question is about specific documents, policies,
    reports, manuals, or any content that has been uploaded to the system.
    """
    if not settings.RAG_ENABLED:
        return "Document search is not enabled."

    # Check if any documents have been ingested
    stats = get_collection_stats()
    if stats.get("total_chunks", 0) == 0:
        return (
            "No documents have been ingested yet. "
            "Run: uv run python rag/ingest.py --path ./my_docs/"
        )

    results = retrieve(query)
    if not results:
        return "No relevant documents found for this query."

    return f"Relevant document excerpts:\n\n{results}"

# ── Tool registry ──────────────────────────────────────────────────────────────
# Add or remove tools here — the agent picks them up automatically
ALL_TOOLS = [
    web_search,
    get_current_time,
    calculator,
    http_get,
    search_documents,    # ← add this
]

TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}