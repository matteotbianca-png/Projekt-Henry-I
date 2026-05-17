"""Tavily web search tool — gives Henry access to current information from the web."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SEARCH_HINT_PATTERNS: tuple[str, ...] = (
    "search",
    "look up",
    "find out",
    "what is happening",
    "latest",
    "current",
    "today",
    "news",
    "who won",
    "weather",
    "price of",
    "how much does",
    "when is",
    "where is",
    "what happened",
    "recent",
)


def _looks_like_search_query(text: str) -> bool:
    """Cheap heuristic: does the user text suggest a web lookup might help?"""
    lower = text.lower()
    return any(p in lower for p in _SEARCH_HINT_PATTERNS)


class HenryWebSearch:
    """Thin wrapper around LangChain's TavilySearchResults for Henry's tool system."""

    def __init__(self, *, max_results: int = 3) -> None:
        self._max_results = max_results
        self._tool: Any | None = None
        self._init_attempted = False
        self._api_key = os.environ.get("TAVILY_API_KEY", "").strip()

        if not self._api_key:
            logger.warning("TAVILY_API_KEY not set; web search disabled.")

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def _ensure_tool(self) -> bool:
        if self._tool is not None:
            return True
        if self._init_attempted or not self._api_key:
            return False

        self._init_attempted = True
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults

            self._tool = TavilySearchResults(
                max_results=self._max_results,
                tavily_api_key=self._api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("TavilySearchResults init failed: %s", exc)
            return False
        return True

    def query(self, search_text: str) -> str:
        """Run a web search and return a formatted context block (empty string on failure)."""
        if not self.available or not search_text.strip():
            return ""
        if not self._ensure_tool():
            return ""
        try:
            results: list[dict[str, Any]] = self._tool.invoke(search_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily search failed: %s", exc)
            return ""
        if not results:
            return ""
        lines: list[str] = []
        for hit in results:
            url = hit.get("url", "")
            content = hit.get("content", "").strip()
            if content:
                lines.append(f"- {content}" + (f" ({url})" if url else ""))
        if not lines:
            return ""
        return "Web search results (use to inform your answer):\n" + "\n".join(lines)

    def search_if_relevant(self, user_text: str) -> str:
        """Only run a search when the user text hints at needing live information."""
        if not self.available:
            return ""
        if not _looks_like_search_query(user_text):
            return ""
        return self.query(user_text)
