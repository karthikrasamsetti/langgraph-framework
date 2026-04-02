import time
import logging
import sys
import structlog
from functools import wraps
from config.settings import get_settings

settings = get_settings()

# ── Windows ANSI fix ───────────────────────────────────────────────────────────
if sys.platform == "win32":
    import ctypes
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass   # not a real terminal (e.g. PyCharm) — skip silently


# ── Shared processors ──────────────────────────────────────────────────────────
shared_processors = [
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="%H:%M:%S"),   # short time for dev
    structlog.processors.StackInfoRenderer(),
    structlog.processors.ExceptionRenderer(),
]

IS_DEV = settings.LOG_LEVEL.upper() == "DEBUG"

# ── Configure structlog ────────────────────────────────────────────────────────
structlog.configure(
    processors=shared_processors + [
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# ── Choose renderer ────────────────────────────────────────────────────────────
if IS_DEV:
    renderer = structlog.dev.ConsoleRenderer(
        colors=True,
        exception_formatter=structlog.dev.plain_traceback,
        sort_keys=False,
        pad_event=40,          # aligns key=value pairs nicely
    )
else:
    renderer = structlog.processors.JSONRenderer()

# ── Attach to stdlib root handler ─────────────────────────────────────────────
formatter = structlog.stdlib.ProcessorFormatter(
    processor=renderer,
    foreign_pre_chain=shared_processors,
)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.handlers.clear()
root_logger.addHandler(handler)
root_logger.setLevel(
    getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
)

log = structlog.get_logger("langgraph")


# ── Token tracker ──────────────────────────────────────────────────────────────
class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def record(self, response):
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            self.prompt_tokens     += response.usage_metadata.get("input_tokens", 0)
            self.completion_tokens += response.usage_metadata.get("output_tokens", 0)
            self.calls += 1

    @property
    def total_tokens(self):
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        return {
            "calls":             self.calls,
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
        }


# ── Node timing decorator ──────────────────────────────────────────────────────
def timed_node(node_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            start = time.perf_counter()
            log.info("node_start", node=node_name,
                     iteration=state.get("iteration_count", 0))
            try:
                result = fn(state, *args, **kwargs)
                log.info(
                    "node_complete",
                    node=node_name,
                    latency_ms=round((time.perf_counter() - start) * 1000, 1),
                )
                return result
            except Exception as exc:
                log.error("node_error", node=node_name, error=str(exc))
                raise
        return wrapper
    return decorator