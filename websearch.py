"""Web search for questions about current events, prices, news, etc.

Tavily is used first when TAVILY_API_KEY is set in .env.
If there is no key, or Tavily fails / runs out of credits, DuckDuckGo (ddgs) is used.
DuckDuckGo needs no key.
"""
from __future__ import annotations

import logging
import os
import re

import httpx

logger = logging.getLogger("document-chatbot.web")

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = int(os.getenv("WEB_RESULTS", "5"))
AUTO_SEARCH = os.getenv("WEB_SEARCH_AUTO", "true").lower() in {"1", "true", "yes", "on"}

# Words that usually mean "this needs up-to-date information".
_FRESH = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|this (week|month|year)|latest|newest|recent|recently|"
    r"current|currently|right now|news|headlines?|price|prices|cost|rate|rates|stock|share price|"
    r"weather|forecast|score|scores|match|who won|winner|results?|election|released?|launch(ed)?|"
    r"update|trending|20[2-3]\d)\b",
    re.I,
)


def needs_web(question: str) -> bool:
    """Guess whether a question needs fresh information from the web."""
    return AUTO_SEARCH and bool(_FRESH.search(question))


def _tavily(query: str) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        raise RuntimeError("no Tavily key")
    response = httpx.post(
        TAVILY_URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": MAX_RESULTS, "search_depth": "basic"},
        timeout=15,
    )
    response.raise_for_status()
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in response.json().get("results", [])
    ]


def _duckduckgo(query: str) -> list[dict]:
    from ddgs import DDGS
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")}
        for r in DDGS().text(query, max_results=MAX_RESULTS)
    ]


def search(query: str) -> tuple[list[dict], str]:
    """Return (results, engine name). Results are [{title, url, content}, ...]."""
    for name, engine in (("Tavily", _tavily), ("DuckDuckGo", _duckduckgo)):
        try:
            results = [r for r in engine(query) if r["url"]]
            if results:
                logger.info("Web search via %s: %s results", name, len(results))
                return results, name
        except Exception as error:
            if str(error) != "no Tavily key":
                logger.warning("%s search failed: %s", name, error)
    return [], ""