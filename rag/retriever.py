"""
Retriever — searches ChromaDB for chunks relevant to a query.
Called by the search_documents tool in registry.py.
"""
from functools import lru_cache
from config.settings import get_settings
from rag.embeddings import get_embedding_function

settings = get_settings()


@lru_cache(maxsize=1)
def get_collection():
    """
    Returns the ChromaDB collection.
    Cached — connects once per process.
    """
    import chromadb

    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_or_create_collection(
        name=settings.VECTOR_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},   # cosine similarity — best for text
    )
    return collection


def retrieve(query: str, top_k: int = None) -> str:
    """
    Searches the vector DB for chunks relevant to the query.
    Returns formatted text ready to inject into the LLM prompt.

    Returns empty string if no documents have been ingested yet.
    """
    k = top_k or settings.RAG_TOP_K

    try:
        collection = get_collection()

        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            return ""

        results = collection.query(
            query_texts=[query],
            n_results=min(k, count),   # can't request more than what exists
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not documents:
            return ""

        # Format chunks with source info
        formatted = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            source   = meta.get("source", "unknown")
            filename = meta.get("filename", source)
            page     = meta.get("page", "")
            page_str = f" (page {page})" if page else ""
            score    = round((1 - dist) * 100, 1)   # convert distance to similarity %

            formatted.append(
                f"[Source {i+1}: {filename}{page_str} — {score}% match]\n{doc}"
            )

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Retrieval error: {e}"


def get_collection_stats() -> dict:
    """Returns basic stats about what's been ingested."""
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "total_chunks": count,
            "collection":   settings.VECTOR_COLLECTION,
            "vector_db":    settings.VECTOR_DB_PATH,
        }
    except Exception as e:
        return {"error": str(e)}