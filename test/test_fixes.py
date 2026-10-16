"""Tests for the 3 fixes: tables, <br> line breaks, and off-topic answers.

Run:  python -m pytest -q
No real AI is called. The markdown tests need Node.js (they skip if it's missing).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Safe test settings BEFORE main.py is imported: fake key, no login, temp database.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ["AUTH_ENABLED"] = "false"
os.environ["WEB_SEARCH_AUTO"] = "false"
os.environ["DB_PATH"] = str(Path(tempfile.mkdtemp()) / "test_chats.db")

import main  # noqa: E402


# ---------------------------------------------------------------- Fix 3: follow-ups
@pytest.mark.parametrize("question", ["whats ai", "what's AI", "what is machine learning", "capital of France"])
def test_new_questions_are_not_follow_ups(question):
    assert main.is_follow_up(question) is False


@pytest.mark.parametrize("question", ["explain more", "why is that?", "tell me more", "what about Java?", "simplify it"])
def test_real_follow_ups_are_detected(question):
    assert main.is_follow_up(question) is True


def test_long_message_is_never_a_follow_up():
    assert main.is_follow_up("explain how neural networks learn from data step by step please") is False


def _searched_query(monkeypatch, history, question):
    """Run build_messages and return the text that was sent to the document search."""
    seen = {}

    def fake_search(query, **kwargs):
        seen["query"] = query
        return []

    monkeypatch.setattr(main, "get_history", lambda session_id: history)
    monkeypatch.setattr(main.rag, "hybrid_search", fake_search)
    main.build_messages(question, "test-session")
    return seen["query"]


FRUIT_CHAT = [
    {"role": "user", "content": "Make a table of 3 fruits with benefits"},
    {"role": "assistant", "content": "| Fruit | Benefits |"},
]


def test_new_short_question_is_searched_alone(monkeypatch):
    assert _searched_query(monkeypatch, FRUIT_CHAT, "whats ai") == "whats ai"


def test_follow_up_is_searched_with_previous_question(monkeypatch):
    query = _searched_query(monkeypatch, FRUIT_CHAT, "explain more")
    assert "fruits" in query and "explain more" in query


def test_system_prompt_rules():
    prompt = main.SYSTEM_PROMPT
    assert '"AI" = artificial intelligence' in prompt
    assert "ignore them silently" in prompt
    assert "Never use HTML tags" in prompt


# ---------------------------------------------------------------- Fixes 1 & 2: markdown (JS)
NODE = shutil.which("node")


def render(markdown: str) -> str:
    """Run the real renderMarkdown() from static/index.html in Node and return the HTML."""
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    start = html.index("const escapeHtml")
    end = html.index("/* ================= History list")
    script = html[start:end] + f"\nprocess.stdout.write(renderMarkdown({json.dumps(markdown)}));"
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True, encoding="utf-8", check=True)
    return result.stdout


needs_node = pytest.mark.skipif(NODE is None, reason="Node.js not installed")


@needs_node
def test_table_renders_as_html_table():
    out = render("| Name | Use |\n|---|---|\n| Python | AI |\n| Java | Apps |")
    assert "<table>" in out and "<th>Name</th>" in out and "<td>Java</td>" in out
    assert "|" not in out  # no raw pipes left


@needs_node
def test_br_becomes_real_line_break():
    out = render("| Fruit | Benefits |\n|---|---|\n| Apple | Fiber<br>Vitamins<br/>Heart |")
    assert "&lt;br" not in out  # no visible <br> text
    assert "Fiber<br>Vitamins<br>Heart" in out


@needs_node
def test_other_html_is_still_escaped():
    out = render("<script>alert(1)</script>")
    assert "<script>" not in out and "&lt;script&gt;" in out


@needs_node
def test_code_block_still_works():
    out = render("```python\nprint('hi')\n```")
    assert "<pre><code>" in out and "print(" in out