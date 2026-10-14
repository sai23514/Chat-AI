"""Local document search (RAG) shared by every AI provider.

ingest.py calls build_index() to read ./docs, split it into chunks, embed them
on your own computer (no API key needed) and save them in ./index.
main.py calls search() to find the chunks that best match a question.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger("document-chatbot.rag")

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".htm"}

_embedder = None
_chunks: list[dict] | None = None
_vectors: np.ndarray | None = None


def embedder():
    """Load the local embedding model once (downloads ~70 MB on first use)."""
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


# ---------------------------------------------------------------- reading files
def read_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return [(page_number or None, text), ...] for one file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if suffix == ".docx":
        import docx
        document = docx.Document(str(path))
        return [(None, "\n".join(p.text for p in document.paragraphs))]
    text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", text, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    return [(None, text)]


def split_text(text: str) -> list[str]:
    """Split into ~CHUNK_CHARS pieces, preferring paragraph and sentence breaks."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > CHUNK_CHARS * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


# ---------------------------------------------------------------- build / load
def build_index() -> int:
    if not DOCS_DIR.is_dir():
        raise SystemExit(f"Documents folder not found: {DOCS_DIR.resolve()}")
    files = sorted(p for p in DOCS_DIR.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
    if not files:
        raise SystemExit(f"No supported files in {DOCS_DIR.resolve()} ({', '.join(sorted(SUPPORTED))})")

    chunks: list[dict] = []
    for path in files:
        try:
            pages = read_pages(path)
        except Exception as error:
            print(f"  ! Skipped {path.name}: {error}")
            continue
        count = 0
        for page, text in pages:
            for piece in split_text(text):
                chunks.append({"text": piece, "file": path.name, "page": page})
                count += 1
        print(f"  {path.name}: {count} chunks")

    if not chunks:
        raise SystemExit("No text could be extracted. Scanned PDFs need OCR first.")

    print(f"Embedding {len(chunks)} chunks locally with {EMBED_MODEL} ...")
    vectors = np.array(list(embedder().embed([c["text"] for c in chunks])), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    return len(chunks)

def add_file(path: Path) -> int:
    """Add one uploaded file to the existing index without rebuilding everything."""
    global _chunks, _vectors
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    has_index()  # make sure the current index is loaded
    new = [{"text": piece, "file": path.name, "page": page}
           for page, text in read_pages(path) for piece in split_text(text)]
    if not new:
        raise ValueError("No text could be extracted from this file.")
    vecs = np.array(list(embedder().embed([c["text"] for c in new])), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    # Drop old chunks of a file with the same name, then append the new ones.
    keep = [i for i, c in enumerate(_chunks or []) if c["file"] != path.name]
    old_vecs = _vectors[keep] if _vectors is not None and keep else np.empty((0, vecs.shape[1]), np.float32)
    _chunks = [_chunks[i] for i in keep] + new
    _vectors = np.vstack([old_vecs, vecs])
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", _vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(_chunks, ensure_ascii=False), encoding="utf-8")
    return len(new)

def load_index() -> bool:
    global _chunks, _vectors
    vec_path, chunk_path = INDEX_DIR / "vectors.npy", INDEX_DIR / "chunks.json"
    if not (vec_path.is_file() and chunk_path.is_file()):
        logger.warning("No document index found. Run: python ingest.py")
        _chunks, _vectors = [], None
        return False
    _vectors = np.load(vec_path)
    _chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
    logger.info("Loaded %s document chunks", len(_chunks))
    return True


def has_index() -> bool:
    if _chunks is None:
        load_index()
    return bool(_chunks)


def search(query: str, k: int = 5, min_score: float = 0.35) -> list[dict]:
    """Return the k best chunks as dicts with text, file, page, score."""
    if not has_index():
        return []
    q = np.array(next(iter(embedder().query_embed(query))), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    scores = _vectors @ q
    best = np.argsort(-scores)[:k]
    top = float(scores[best[0]]) if len(best) else 0.0
    # Keep chunks that pass the minimum and are close to the best match.
    return [
        {**_chunks[i], "score": float(scores[i])}
        for i in best if scores[i] >= min_score and scores[i] >= top - 0.15
    ]


def source_label(chunk: dict) -> str:
    return f"{chunk['file']} (p. {chunk['page']})" if chunk.get("page") else chunk["file"]
