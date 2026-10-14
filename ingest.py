"""Build the local document index from ./docs.

Run this whenever you add, change or remove files in docs/:
    python ingest.py
It rebuilds the index from scratch each time, so there are no duplicates.
No API key is needed: embeddings run on your computer.
"""
from dotenv import load_dotenv

load_dotenv()
import rag  # noqa: E402

if __name__ == "__main__":
    print(f"Reading documents from {rag.DOCS_DIR.resolve()} ...")
    total = rag.build_index()
    print(f"\nDone. {total} chunks saved in {rag.INDEX_DIR.resolve()}")
    print("Restart the server (uvicorn main:app --reload) to use the new index.")
