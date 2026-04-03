import sqlite3
from functools import lru_cache
from config.settings import get_settings
from observability.logger import log

# ── Keep connection alive at module level ──────────────────────────────────────
# SQLite connection must stay open for the lifetime of the process.
# Storing it at module level prevents it from being garbage collected.
_sqlite_conn = None


@lru_cache()
def get_checkpointer():
    global _sqlite_conn
    s = get_settings()
    log.info("checkpointer_init", type=s.CHECKPOINTER)

    if s.CHECKPOINTER == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    elif s.CHECKPOINTER == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            # Open connection manually and store at module level
            # check_same_thread=False required for FastAPI's thread pool
            _sqlite_conn = sqlite3.connect(
                s.SQLITE_DB_PATH,
                check_same_thread=False
            )
            saver = SqliteSaver(_sqlite_conn)
            log.info("sqlite_connected", path=s.SQLITE_DB_PATH)
            return saver

        except ImportError:
            log.warning("sqlite_unavailable",
                        msg="Install: uv pip install langgraph-checkpoint-sqlite")
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

        except Exception as e:
            log.warning("sqlite_fallback", error=str(e))
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    elif s.CHECKPOINTER == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            return PostgresSaver.from_conn_string(s.POSTGRES_CONNECTION_STRING)
        except Exception as e:
            log.warning("postgres_fallback", error=str(e))
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()

    raise ValueError(f"Unknown CHECKPOINTER: {s.CHECKPOINTER}")