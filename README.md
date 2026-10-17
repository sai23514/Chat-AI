# Chat AI
live link- https://chat-ai.malaysiawest.cloudapp.azure.com/

A general-purpose AI chat app I built from scratch with FastAPI. You can just chat with it like any assistant, or upload your own files and ask questions about them. Behind the scenes it talks to several free AI providers, and when one is busy or out of quota, the next one picks up automatically, so the chat keeps working.

It started as a small "ask questions about a PDF" project and grew, step by step, into a full chat app with accounts, web search, voice input, image understanding, exports, and a live deployment on a cloud server.

---

## What it can do

- **Chat about anything**: coding, writing, explanations, everyday questions.
- **Use several AI providers with automatic failover**: Groq, OpenRouter, Mistral, Cloudflare Workers AI, SambaNova, Cohere and Google Gemini. If one returns a rate limit (429) or is down, it rests for a while and the next provider answers. You can also pick a specific model from the model picker.
- **Answer from your own documents**: upload PDF, DOCX, TXT, MD, CSV, JSON or HTML files, and the app finds the relevant parts and answers from them, naming the file it used.
- **Read scanned PDFs and images**: built-in OCR pulls text out of scanned pages and pictures (PNG, JPG, WEBP, BMP, TIFF).
- **Understand images**: attach up to 3 images to a message, and vision-capable models describe or reason about them.
- **Search the web**: a globe button turns on live web search. It also switches on automatically for questions about news, prices, weather, scores and other fresh topics. Sources are cited as links.
- **Take voice input**: press the mic button, speak, and your words are transcribed into the message box.
- **Stream replies**: answers appear word by word, as they're generated.
- **Handle accounts and privacy**: sign up and log in. Every user sees only their own chats.
- **Keep history**: past chats are grouped by date in the sidebar (Today, Yesterday, Previous 7 days, and so on).
- **Export chats**: save any chat as Markdown, plain text, JSON or PDF.
- **Format answers properly**: tables, code blocks, lists, headings, quotes and links all render cleanly.
- **Light and dark themes**, with a glassy, modern UI.

---

## Tech stack

| Part | What I used |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| AI providers | OpenAI Python SDK pointed at each provider's OpenAI-compatible endpoint |
| Document search | fastembed (local embeddings), BM25 keyword search, a cross-encoder reranker, NumPy |
| File reading | pypdf, python-docx, pypdfium2 |
| OCR | RapidOCR + ONNX Runtime |
| Web search | Tavily (if a key is set), with DuckDuckGo as a free fallback |
| Voice | Groq Whisper speech-to-text |
| Database | SQLite (chat history, users, sessions) |
| Frontend | A single HTML file with plain JavaScript and CSS, no framework |
| Hosting | Ubuntu VM on Azure, systemd, Caddy (automatic HTTPS) |
| Tests | pytest |

---

## How a message flows

```
You type a message
      |
      v
FastAPI checks you're logged in and that the chat is yours
      |
      v
Document search (only if you uploaded files)
  - meaning search (embeddings) + keyword search (BM25)
  - results merged, then re-scored by a reranker
  - weak matches are dropped
      |
      v
Web search (if the globe is on, or the question needs fresh info)
      |
      v
Prompt = system rules + recent chat history + excerpts/results + your message
      |
      v
Provider 1 --(429 / error)--> Provider 2 --> Provider 3 ...
      |
      v
Reply streams back to the browser and is saved to SQLite
```

A detail I'm happy with: a short follow-up like "explain more" gets searched together with your previous question, so the search knows what "more" refers to. A short new question like "what's AI" gets searched on its own, so an old topic doesn't leak into the answer.

---

## Project structure

```
chatbot_upgrade/
  main.py          FastAPI app: chat, streaming, uploads, voice, provider failover
  auth.py          Sign up, login, logout, sessions, chat ownership
  rag.py           Document reading, chunking, hybrid search and reranking
  ocr.py           OCR for scanned PDFs and images
  websearch.py     Tavily / DuckDuckGo web search
  ingest.py        Optional: build the index from a local docs/ folder
  chat.py          Optional: chat from the terminal
  test.py          Quick check of API keys and the index
  static/
    index.html     The whole frontend (HTML + CSS + JS)
  test/
    test_fixes.py  pytest tests
  requirements.txt
  .env.example     Template for settings (no real keys)
```

Created when you run it (and ignored by git): `.env`, `chats.db`, `uploads/`, `index/`, `venv/`.

---

## Getting started (local)

### 1. Clone and set up Python

```bash
git clone <your-repo-url>
cd chatbot_upgrade
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

### 2. Add your settings

Copy `.env.example` to `.env` and add at least **one** provider key. The more keys you add, the better the failover works. All of these providers have free tiers.

```env
GROQ_API_KEY=your-key-here
OPENROUTER_API_KEY=your-key-here
# ...add any others you have
```

### 3. Run it

```bash
python -m uvicorn main:app --reload --port 8001
```

Open `localhost:8001` in your browser, create an account, and start chatting.

> Tip: run it with `python -m uvicorn` rather than plain `uvicorn`. That way it always uses the virtual environment's packages.

The first time you upload a document, the app downloads a small embedding model (about 70 MB). After that, it's cached.

---

## Configuration

Everything is configured through `.env`. Here are the main settings. Only one provider key is required.

### AI providers

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Groq (also used for voice transcription) |
| `OPENROUTER_API_KEY` | OpenRouter |
| `MISTRAL_API_KEY` | Mistral |
| `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID` | Cloudflare Workers AI |
| `SAMBANOVA_API_KEY` | SambaNova |
| `COHERE_API_KEY` | Cohere |
| `GEMINI_API_KEY` | Google Gemini |
| `PROVIDER_ORDER` | The order to try providers in Auto mode, e.g. `groq,openrouter,mistral,cloudflare,sambanova,cohere,gemini` |
| `<PROVIDER>_MODEL` | Override a provider's default model, e.g. `GROQ_MODEL` |
| `PROVIDER_COOLDOWN_SECONDS` | How long a provider rests after a 429 (default 60) |

### Accounts

| Variable | Purpose |
|---|---|
| `AUTH_ENABLED` | Turn login on or off (on by default) |
| `ALLOW_SIGNUP` | Allow new accounts. I set this to `false` on the live server after creating my own account. |
| `COOKIE_SECURE` | `true` when served over HTTPS |
| `SESSION_DAYS` | How long a login lasts |

### Documents and search

| Variable | Purpose |
|---|---|
| `MAX_UPLOAD_MB` | Upload size limit (default 20) |
| `TOP_K`, `MIN_SCORE` | How many excerpts to use, and how relevant they must be |
| `HYBRID_SEARCH`, `RERANK` | Turn the keyword search and the reranker on or off |
| `OCR_ENABLED` | OCR for scanned files and images |
| `TAVILY_API_KEY` | Optional. Better web search (DuckDuckGo is used without it). |
| `WEB_SEARCH_AUTO` | Automatically search the web for "fresh" questions |

**Never commit `.env`.** It's already in `.gitignore`.

---

## Running the tests

```bash
python -m pip install pytest
python -m pytest -q
```

The tests never call a real AI and never touch your real chats. They use a temporary database. They check:

- markdown tables render as real tables
- `<br>` from the AI becomes a real line break, while unsafe HTML is still blocked
- follow-up detection ("explain more" vs "what's AI")
- the system prompt rules

The markdown tests run the real frontend code through Node.js, and they skip if Node isn't installed.

---

## Deployment

The app runs on a small Ubuntu 24.04 virtual machine on Azure (an Azure for Students subscription). The setup:

1. **VM**: a small burstable Linux VM with an SSD, and only ports 22 (SSH), 80 and 443 open. I added a 2 GB swap file to give the embedding and OCR models some breathing room.
2. **App**: the code lives in the home folder with its own virtual environment. A **systemd** service runs Uvicorn on `127.0.0.1:8001`, so it starts on boot, restarts if it crashes, and keeps running after I close SSH. The app port is never exposed to the internet directly.
3. **HTTPS**: **Caddy** sits in front as a reverse proxy, with a free Azure DNS name. Caddy gets and renews the HTTPS certificate automatically. Because the chat streams its replies, Caddy is set to flush immediately (`flush_interval -1`), so words appear as they arrive instead of all at once.
4. **Production settings**: `COOKIE_SECURE=true`, and sign-up turned off after the owner account was created.

### Updating the live app

```bash
# on my PC
git push
git archive -o app.zip HEAD
scp -i <path-to-ssh-key> app.zip <user>@<server>:~

# on the server
unzip -o ~/app.zip -d ~/chatbot_upgrade
sudo systemctl restart chatai
```

`git archive` packs only committed code, so the server's `.env`, database and uploads are never overwritten.

---

## Security notes

- Passwords are hashed with **scrypt**. Plain passwords are never stored.
- Logins use an **httpOnly** session cookie. Only a SHA-256 hash of each session token is stored in the database.
- After 5 wrong passwords, logins are paused for 15 minutes.
- Every chat endpoint checks that the chat belongs to the logged-in user.
- Text from documents and web pages is treated as source material, never as instructions to the AI, which guards against prompt injection.
- AI output is HTML-escaped before rendering, so a reply can't inject scripts into the page.
- API keys live only in `.env` on the machine running the app.

One thing worth knowing: free AI tiers may use prompts to improve their models, so I don't send anything sensitive through them.

---

## What I learned

- Building **RAG** properly: chunking, embeddings, why hybrid search (meaning + keywords) beats either one alone, and how a reranker cleans up the results.
- Designing **failover** across providers that all speak the OpenAI API format, and handling rate limits gracefully.
- **Streaming** responses end to end (NDJSON from FastAPI to the browser), and why the reverse proxy needs to flush immediately.
- **Auth from scratch**: password hashing, session tokens, cookies, brute-force protection.
- Real **deployment**: Linux servers, SSH keys, systemd, reverse proxies, HTTPS, and keeping secrets out of git.
- Small things that matter, like making sure "what's AI" means *artificial intelligence* and not something from an unrelated document.

---

## Roadmap

- [ ] Per-user daily limits and an invite code, so sign-up can be reopened safely
- [ ] Nightly backups of the database and uploads to cloud storage
- [ ] More security hardening (CSRF protection, security headers)
- [ ] More tests, plus GitHub Actions to run them on every push
- [ ] Server-side chat history (possibly PostgreSQL)
- [ ] Summarising long chats so they fit in the context window
- [ ] Usage logging and automatic cleanup of old uploads
- [ ] App-issued API keys with an OpenAI-compatible endpoint

---

## Acknowledgements

Thanks to the teams behind FastAPI, fastembed, RapidOCR and Caddy, and to the AI providers whose free tiers made this project possible to build and run as a student.
