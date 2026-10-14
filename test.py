"""Check every configured provider and the document index.

    python test.py
Shows which keys work, how fast each provider answers, and what the index finds.
"""
import time

from main import PROVIDERS
import rag

QUESTION = "What is EPF? Answer in two lines."


def test_provider(provider) -> None:
    print(f"\n=== {provider.name} | {provider.model} ===")
    start = time.perf_counter()
    first = None
    try:
        stream = provider.client.chat.completions.create(
            model=provider.model, stream=True,
            messages=[{"role": "user", "content": QUESTION}], **provider.extra,
        )
        text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter() - start
                text += chunk.choices[0].delta.content
        print(text.strip()[:300])
        print(f"OK  first text {first:.2f}s, total {time.perf_counter() - start:.2f}s"
              if first else "OK but no text returned")
    except Exception as error:
        code = getattr(error, "status_code", "")
        print(f"FAILED {code} {type(error).__name__}: {str(error)[:200]}")


if __name__ == "__main__":
    for p in PROVIDERS:
        test_provider(p)
    print("\n=== Document index ===")
    if rag.has_index():
        for hit in rag.search(QUESTION, k=3, min_score=0):
            print(f"{hit['score']:.2f}  {rag.source_label(hit)}  {hit['text'][:80]!r}")
    else:
        print("No index yet. Run: python ingest.py")
