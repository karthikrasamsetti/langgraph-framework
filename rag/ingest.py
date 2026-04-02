"""
Ingestion script — loads documents into ChromaDB.
Run from command line:

  uv run python rag/ingest.py --path ./my_docs/
  uv run python rag/ingest.py --path ./my_docs/report.pdf
  uv run python rag/ingest.py --url https://example.com/page
  uv run python rag/ingest.py --clear        # wipe the vector DB

Supports: PDF, TXT, DOCX, MD, URLs
"""
import os
import sys
import uuid
import argparse
from pathlib import Path
from typing import List, Dict

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from rag.embeddings import get_embedding_function

settings = get_settings()


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_pdf(path: str) -> List[Dict]:
    """Extract text from PDF — one entry per page."""
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "text":     text.strip(),
                "source":   path,
                "filename": Path(path).name,
                "page":     i + 1,
                "type":     "pdf",
            })
    print(f"  Extracted {len(pages)} pages from {Path(path).name}")
    return pages


def extract_txt(path: str) -> List[Dict]:
    """Extract text from plain text or markdown file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if not text.strip():
        return []
    print(f"  Extracted text from {Path(path).name}")
    return [{
        "text":     text.strip(),
        "source":   path,
        "filename": Path(path).name,
        "page":     "",
        "type":     "txt",
    }]


def extract_docx(path: str) -> List[Dict]:
    """Extract text from Word document."""
    from docx import Document
    doc   = Document(path)
    text  = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not text.strip():
        return []
    print(f"  Extracted text from {Path(path).name}")
    return [{
        "text":     text.strip(),
        "source":   path,
        "filename": Path(path).name,
        "page":     "",
        "type":     "docx",
    }]


def extract_url(url: str) -> List[Dict]:
    """Extract text from a webpage."""
    import requests
    from bs4 import BeautifulSoup

    print(f"  Fetching {url}...")
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup  = BeautifulSoup(resp.text, "html.parser")

    # Remove nav, footer, scripts, styles — keep main content
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    # Collapse excessive blank lines
    lines = [l for l in text.splitlines() if l.strip()]
    text  = "\n".join(lines)

    if not text.strip():
        return []

    from urllib.parse import urlparse
    domain = urlparse(url).netloc

    print(f"  Extracted {len(text)} chars from {domain}")
    return [{
        "text":     text[:50_000],   # cap at 50k chars per page
        "source":   url,
        "filename": domain,
        "page":     "",
        "type":     "url",
    }]


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Splits text into overlapping chunks.
    overlap means each chunk shares some text with the next —
    this prevents answers being cut off at chunk boundaries.
    """
    chunks = []
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap   # step back by overlap amount

    return chunks


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest_documents(pages: List[Dict]) -> int:
    """
    Chunks all pages and stores them in ChromaDB.
    Returns total number of chunks stored.
    """
    import chromadb

    client     = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    collection = client.get_or_create_collection(
        name=settings.VECTOR_COLLECTION,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks    = []
    all_metadatas = []
    all_ids       = []

    for page in pages:
        chunks = chunk_text(
            page["text"],
            chunk_size=settings.RAG_CHUNK_SIZE,
            overlap=settings.RAG_CHUNK_OVERLAP,
        )
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "source":   page["source"],
                "filename": page["filename"],
                "page":     str(page.get("page", "")),
                "type":     page["type"],
            })
            all_ids.append(str(uuid.uuid4()))

    if not all_chunks:
        print("  No chunks to store.")
        return 0

    # ChromaDB has a batch size limit — process in batches of 100
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i:i+batch_size],
            metadatas=all_metadatas[i:i+batch_size],
            ids=all_ids[i:i+batch_size],
        )

    print(f"  Stored {len(all_chunks)} chunks in ChromaDB")
    return len(all_chunks)


def clear_collection():
    """Wipe all documents from the vector DB."""
    import chromadb
    client = chromadb.PersistentClient(path=settings.VECTOR_DB_PATH)
    client.delete_collection(settings.VECTOR_COLLECTION)
    print(f"Cleared collection: {settings.VECTOR_COLLECTION}")


def process_path(path: str) -> List[Dict]:
    """Auto-detects file type and extracts text."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext in (".txt", ".md"):
        return extract_txt(path)
    elif ext == ".docx":
        return extract_docx(path)
    else:
        print(f"  Skipping unsupported file type: {ext}")
        return []


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG vector DB")
    parser.add_argument("--path",  type=str, help="File or folder path to ingest")
    parser.add_argument("--url",   type=str, help="URL to ingest")
    parser.add_argument("--clear", action="store_true", help="Clear the vector DB")
    parser.add_argument("--stats", action="store_true", help="Show collection stats")
    args = parser.parse_args()

    if args.clear:
        clear_collection()
        return

    if args.stats:
        from rag.retriever import get_collection_stats
        stats = get_collection_stats()
        print(f"Collection : {stats.get('collection')}")
        print(f"Chunks     : {stats.get('total_chunks')}")
        print(f"Vector DB  : {stats.get('vector_db')}")
        return

    pages = []

    if args.path:
        p = Path(args.path)
        if p.is_dir():
            print(f"Scanning folder: {p}")
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".pdf",".txt",".md",".docx"):
                    print(f"Loading: {f.name}")
                    pages.extend(process_path(str(f)))
        elif p.is_file():
            print(f"Loading: {p.name}")
            pages.extend(process_path(str(p)))
        else:
            print(f"Path not found: {args.path}")
            return

    if args.url:
        pages.extend(extract_url(args.url))

    if not pages:
        print("No content extracted. Check your --path or --url.")
        return

    print(f"\nIngesting {len(pages)} pages...")
    total = ingest_documents(pages)
    print(f"\nDone — {total} chunks stored in {settings.VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()