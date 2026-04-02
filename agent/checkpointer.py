from functools import lru_cache
from config.settings import get_settings
from observability.logger import log


@lru_cache()
def get_checkpointer():
    s = get_settings()
    log.info("checkpointer_init", type=s.CHECKPOINTER)

    if s.CHECKPOINTER == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    elif s.CHECKPOINTER == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver.from_conn_string(s.SQLITE_DB_PATH)

    elif s.CHECKPOINTER == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        return PostgresSaver.from_conn_string(s.POSTGRES_CONNECTION_STRING)

    raise ValueError(f"Unknown CHECKPOINTER: {s.CHECKPOINTER}")