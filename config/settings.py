from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM Provider ──────────────────────────────────────────────
    # Change LLM_PROVIDER in .env — nothing else in your code changes
    LLM_PROVIDER: Literal[
        "openai", "anthropic", "bedrock", "ollama", "huggingface", "azure_openai", "groq" 
    ] = "openai"

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-5-sonnet-20241022"

    # AWS Bedrock
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    # Ollama (local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # HuggingFace / open-source
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"
    HUGGINGFACE_INFERENCE_URL: Optional[str] = None  # self-hosted endpoint

    # Azure OpenAI
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4o"
    AZURE_OPENAI_API_VERSION: str = "2024-08-01-preview"

    # ── Groq ──────────────────────────────────────────────────────────────────────
    GROQ_API_KEY:  Optional[str] = None
    GROQ_MODEL:    str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

    # ── LLM behaviour ─────────────────────────────────────────────
    LLM_TEMPERATURE: float = Field(default=0.0, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2048, ge=1)
    LLM_TIMEOUT: int = 60  # seconds

    # ── Agent behaviour ───────────────────────────────────────────
    MAX_ITERATIONS: int = 10       # safety guard — stop infinite loops
    MAX_RETRIES: int = 3           # per-node retry budget
    RETRY_DELAY: float = 1.0       # seconds between retries

    # ── Persistence ───────────────────────────────────────────────
    CHECKPOINTER: Literal["memory", "sqlite", "postgres"] = "memory"
    SQLITE_DB_PATH: str = "./checkpoints.db"
    POSTGRES_CONNECTION_STRING: Optional[str] = None

    # ── Observability ─────────────────────────────────────────────
    # LOG_LEVEL: str = "INFO"
    # LANGCHAIN_TRACING_V2: bool = False   # set True + add LANGCHAIN_API_KEY for LangSmith
    # LANGCHAIN_API_KEY: Optional[str] = None
    # LANGCHAIN_PROJECT: str = "langgraph-production"

    # ── API ───────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    # ── Observability ─────────────────────────────────────────────
    LOG_LEVEL:                str  = "DEBUG"
    LANGSMITH_TRACING:     bool = False
    LANGSMITH_API_KEY:        Optional[str] = None
    LANGSMITH_PROJECT:        str  = "langgraph-production"
    LANGSMITH_ENDPOINT:       str  = "https://api.smith.langchain.com"   # ← add
    # LANGSMITH_TAGS:           str  = ""                                   # ← add

    # ── Tools ─────────────────────────────────────────────────────────────────────
    TAVILY_API_KEY: Optional[str] = None

    # ── RAG ───────────────────────────────────────────────────────────────────────
    RAG_ENABLED:            bool                    = True
    VECTOR_DB_PATH:         str                     = "./vector_db"
    VECTOR_COLLECTION:      str                     = "documents"
    RAG_CHUNK_SIZE:         int                     = 500
    RAG_CHUNK_OVERLAP:      int                     = 50
    RAG_TOP_K:              int                     = 4

    # ── Embeddings ────────────────────────────────────────────────────────────────
    EMBEDDING_PROVIDER:     Literal["local","openai"] = "local"
    EMBEDDING_LOCAL_MODEL:  str                     = "all-MiniLM-L6-v2"
    EMBEDDING_OPENAI_MODEL: str                     = "text-embedding-3-small"

    # ── Multi-agent ───────────────────────────────────────────────────────────────
    MULTI_AGENT_ENABLED: bool = True
    SUPERVISOR_MAX_ITERATIONS: int = 3   # max specialist calls per query

    # ── Human-in-the-loop ─────────────────────────────────────────────────────────
    HUMAN_IN_LOOP_ENABLED: bool = False
    HUMAN_IN_LOOP_NODES:   str  = "tool_executor,supervisor_run_agent"
    # comma-separated node names to interrupt before

    # ── Auth ──────────────────────────────────────────────────────────────────────
    AUTH_ENABLED:              bool = False
    API_KEYS:                  str  = ""
    # Format: key:name:rpm,key:name:rpm
    # Example: sk-prod-abc:production:100,sk-dev-xyz:development:20

    # ── Rate limiting ─────────────────────────────────────────────────────────────
    RATE_LIMIT_ENABLED:        bool = True
    RATE_LIMIT_WINDOW_SECONDS: int  = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra             = "ignore"    # ← add — silently ignores unknown env vars


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — one parse per process."""
    return Settings()


# ── Helper — outside the class ─────────────────────────────────────────────────
def get_human_in_loop_nodes() -> list[str]:
    """
    Returns HUMAN_IN_LOOP_NODES as a clean Python list.
    Standalone function because Pydantic BaseSettings
    does not support @property decorators.
    """
    s = get_settings()
    if not s.HUMAN_IN_LOOP_ENABLED:
        return []
    return [n.strip() for n in s.HUMAN_IN_LOOP_NODES.split(",") if n.strip()]

def parse_api_keys() -> dict:
    """
    Parses API_KEYS string into a dict.
    Format: key:name:rpm,key:name:rpm
    Returns: {key: {"name": str, "rpm": int}}
    """
    s = get_settings()
    if not s.API_KEYS:
        return {}

    keys = {}
    for entry in s.API_KEYS.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) == 3:
            key, name, rpm = parts
            keys[key.strip()] = {
                "name": name.strip(),
                "rpm":  int(rpm.strip()),
            }
        elif len(parts) == 1:
            # Simple key with no metadata — default 60 rpm
            keys[parts[0].strip()] = {
                "name": "default",
                "rpm":  60,
            }
    return keys