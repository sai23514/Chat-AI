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

import ocr

logger = logging.getLogger("document-chatbot.rag")

DOCS_DIR = Path(os.getenv("DOCS_DIR", "docs"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", "index"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# Step 7: hybrid search (meaning + keywords) and reranking
HYBRID_SEARCH = os.getenv("HYBRID_SEARCH", "true").lower() == "true"
RERANK = os.getenv("RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "-7"))  # unrelated text scores about -11
RERANK_GAP = float(os.getenv("RERANK_GAP", "8"))  # also drop chunks far below the best one
CANDIDATES = int(os.getenv("CANDIDATES", "20"))  # how many chunks each search hands to the reranker
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".html", ".htm"} | ocr.IMAGE_TYPES

_embedder = None
_reranker = None
_tokens: list[list[str]] | None = None  # keyword (BM25) cache, rebuilt when the index changes
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
        raise SystemExit("No text could be extracted, even with OCR.")

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
    global _chunks, _vectors, _tokens
    _chunks, _vectors, _tokens = chunks, vectors, None
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
    global _chunks, _vectors, _tokens
    _tokens = None
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


# ---------------------------------------------------------------- step 7: hybrid
STOPWORDS = set("""a an and are as at be by for from has have how i in is it its of on or
that the this to was were what when where which who why will with you your me my
about does did do can could should would please tell give show much many""".split())


def tokenize(text: str) -> list[str]:
    """Words for keyword search. Codes like C-7 or SSM-2026-0417 are kept whole AND split."""
    words = []
    for word in re.findall(r"[a-z0-9]+(?:[-/.,][a-z0-9]+)*", text.lower()):
        if word not in STOPWORDS:
            words.append(word)
        parts = re.split(r"[-/.,]", word)
        if len(parts) > 1:
            words += [p for p in parts if p and p not in STOPWORDS]
    return words


def keyword_scores(query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """BM25: rare words that appear in a chunk score high; common words barely count."""
    global _tokens
    if _tokens is None:
        _tokens = [tokenize(c["text"]) for c in _chunks]
    n = len(_tokens)
    lengths = np.array([len(t) for t in _tokens], dtype=np.float32)
    avg = float(lengths.mean()) or 1.0
    scores = np.zeros(n, dtype=np.float32)
    for term in set(tokenize(query)):
        tf = np.array([t.count(term) for t in _tokens], dtype=np.float32)
        df = int((tf > 0).sum())
        if not df:
            continue
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        scores += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths / avg))
    return scores


def reranker():
    """Load the reranking model once (downloads ~80 MB on first use)."""
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def hybrid_search(query: str, k: int = 5, min_score: float = 0.6,
                  session_id: str | None = None) -> list[dict]:
    """Meaning search + keyword search, merged, then re-checked by the reranker."""
    if not HYBRID_SEARCH:
        return search(query, k=k, min_score=min_score, session_id=session_id)
    if not has_index():
        return []
    allowed = np.array([not c.get("session") or c.get("session") == session_id for c in _chunks])
    if not allowed.any():
        return []

    # 1) Meaning (vectors). Looser cut than before: the reranker removes the weak ones.
    q = np.array(next(iter(embedder().query_embed(query))), dtype=np.float32)
    q /= np.linalg.norm(q) + 1e-12
    vec = np.where(allowed, _vectors @ q, -1.0)
    vec_floor = min_score - 0.25 if RERANK else min_score
    by_meaning = [i for i in np.argsort(-vec)[:CANDIDATES] if vec[i] >= vec_floor]

    # 2) Keywords (BM25)
    kw = np.where(allowed, keyword_scores(query), 0.0)
    by_keyword = [i for i in np.argsort(-kw)[:CANDIDATES] if kw[i] > 0]

    # 3) Merge with Reciprocal Rank Fusion: high in either list = high overall
    fused: dict[int, float] = {}
    for ranking in (by_meaning, by_keyword):
        for rank, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0.0) + 1 / (60 + rank)
    candidates = sorted(fused, key=fused.get, reverse=True)[:CANDIDATES]
    if not candidates:
        return []

    # 4) Rerank: read question + chunk together for a sharper score
    if RERANK:
        try:
            scores = list(reranker().rerank(query, [_chunks[i]["text"] for i in candidates]))
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            best = float(ranked[0][1])
            return [{**_chunks[i], "score": float(s)} for i, s in ranked[:k]
                    if s >= RERANK_MIN_SCORE and s >= best - RERANK_GAP]
        except Exception as error:
            logger.warning("Reranker failed, using merged results: %s", error)
    return [{**_chunks[i], "score": float(vec[i])} for i in candidates[:k]]


def source_label(chunk: dict) -> str:
    return f"{chunk['file']} (p. {chunk['page']})" if chunk.get("page") else chunk["file"]