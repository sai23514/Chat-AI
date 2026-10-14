Chat AI (multi-provider)

Answers questions from your own documents. Document search runs locally, so
Gemini, Groq and OpenRouter all use the same files. If one provider is busy or
out of quota, the next one answers automatically.

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your three API keys.
3. Put your files in `docs/` (PDF, DOCX, TXT, MD, CSV, JSON, HTML).
4. `python ingest.py` (rebuild any time docs change; first run downloads a ~70 MB embedding model)
5. `python test.py` (checks each key and the index)
6. `uvicorn main:app --reload`, then open http://127.0.0.1:8000/

## How failover works
- Order comes from `PROVIDER_ORDER` (default gemini, groq, openrouter).
- 429 (quota): that provider rests for 60 s and the next one answers.
- 503 (busy): retried once, then the next provider answers.
- Bad key or model: skipped for 5 minutes. The reason is in the server log.
- The chat UI shows which provider answered each message.

## Notes
- Chat history is kept in server memory (resets on restart) and in the browser sidebar.
- Scanned PDFs (images only) need OCR before ingesting.
- Never commit `.env`.
