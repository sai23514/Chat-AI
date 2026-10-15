"""Local document search (RAG) shared by every AI provider.

Shared files: put them in ./docs and run ingest.py (every chat can use them).
Chat files: uploaded with the + button; only that chat can search them.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import numpy as np

import ocr

logger = logging.getLogger("document-chatbot.rag")

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".htm"} | ocr.IMAGE_TYPES

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
        pages, scanned = [], 0
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            # Almost no text = a scanned page (a picture of text), so OCR it.
            if ocr.OCR_ENABLED and len(text.strip()) < ocr.OCR_MIN_CHARS and scanned < ocr.OCR_MAX_PAGES:
                scanned += 1
                try:
                    text = ocr.read_pdf_page(path, i)
                except Exception as error:
                    logger.warning("OCR failed on %s page %s: %s", path.name, i + 1, error)
            pages.append((i + 1, text))
        if scanned:
            logger.info("OCR read %s scanned page(s) in %s", scanned, path.name)
        return pages
    if suffix in ocr.IMAGE_TYPES:
        return [(None, ocr.read_image(path) if ocr.OCR_ENABLED else "")]
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
                chunks.append({"text": piece, "file": path.name, "page": page, "session": ""})
                count += 1
        print(f"  {path.name}: {count} chunks")

    if not chunks:
        raise SystemExit("No text could be extracted. Even OCR found no text.")

    print(f"Embedding {len(chunks)} chunks locally with {EMBED_MODEL} ...")
    vectors = _embed([c["text"] for c in chunks])

    # Keep files that were uploaded in chats; only the shared docs/ part is rebuilt.
    load_index()
    keep = [i for i, c in enumerate(_chunks or []) if c.get("session")]
    if keep and _vectors is not None:
        chunks = [_chunks[i] for i in keep] + chunks
        vectors = np.vstack([_vectors[keep], vectors])
    _save(chunks, vectors)
    return len(chunks)


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(embedder().embed(texts)), dtype=np.float32)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def _save(chunks: list[dict], vectors: np.ndarray) -> None:
    global _chunks, _vectors
    _chunks, _vectors = chunks, vectors
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")


def add_file(path: Path, session_id: str) -> int:
    """Add one file uploaded in a chat. Only that chat (session_id) can search it."""
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    has_index()  # make sure the current index is loaded
    new = [{"text": piece, "file": path.name, "page": page, "session": session_id}
           for page, text in read_pages(path) for piece in split_text(text)]
    if not new:
        raise ValueError("No text could be extracted from this file.")
    vecs = _embed([c["text"] for c in new])
    # Same file uploaded again in the same chat replaces the old copy.
    keep = [i for i, c in enumerate(_chunks or [])
            if not (c.get("session") == session_id and c["file"] == path.name)]
    old = _vectors[keep] if _vectors is not None and keep else np.empty((0, vecs.shape[1]), np.float32)
    _save([_chunks[i] for i in keep] + new, np.vstack([old, vecs]))
    return len(new)


def remove_files(session_id: str, name: str | None = None) -> int:
    """Remove one file (or every file) uploaded in a chat. Returns chunks removed."""
    if not has_index() or not session_id:
        return 0
    def gone(c: dict) -> bool:
        return c.get("session") == session_id and (name is None or c["file"] == name)
    keep = [i for i, c in enumerate(_chunks) if not gone(c)]
    removed = len(_chunks) - len(keep)
    if removed:
        _save([_chunks[i] for i in keep], _vectors[keep] if keep else np.empty((0, _vectors.shape[1]), np.float32))
    return removed


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


def search(query: str, k: int = 5, min_score: float = 0.35, session_id: str | None = None) -> list[dict]:
    """Best chunks from the shared docs/ folder plus this chat's own uploads."""
    if not has_index():
        return []
    allowed = np.array([not c.get("session") or c.get("session") == session_id for c in _chunks])
    if not allowed.any():
        return []
    q = np.array(next(iter(embedder().query_embed(query))), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    scores = np.where(allowed, _vectors @ q, -1.0)
    best = np.argsort(-scores)[:k]
    top = float(scores[best[0]])
    # Keep chunks that pass the minimum and are close to the best match.
    return [
        {**_chunks[i], "score": float(scores[i])}
        for i in best if scores[i] >= min_score and scores[i] >= top - 0.15
    ]


def source_label(chunk: dict) -> str:
    return f"{chunk['file']} (p. {chunk['page']})" if chunk.get("page") else chunk["file"]