import os

# ── Skip HuggingFace network checks on every load ─────────────────────────────
# Model is cached locally after first download — no need to check for updates
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from functools import lru_cache
from config.settings import get_settings

settings = get_settings()


@lru_cache(maxsize=1)
def get_embedding_function():
    provider = settings.EMBEDDING_PROVIDER

    if provider == "local":
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        print(f"[Embeddings] Loading local model: {settings.EMBEDDING_LOCAL_MODEL}")
        return SentenceTransformerEmbeddingFunction(
            model_name=settings.EMBEDDING_LOCAL_MODEL,
            cache_folder="./models_cache",    # use local cache
        )

    elif provider == "openai":
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        print(f"[Embeddings] Using OpenAI: {settings.EMBEDDING_OPENAI_MODEL}")
        return OpenAIEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            model_name=settings.EMBEDDING_OPENAI_MODEL,
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: '{provider}'")