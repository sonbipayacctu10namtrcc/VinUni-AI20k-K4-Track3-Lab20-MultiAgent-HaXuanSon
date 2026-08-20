"""Search client abstraction for ResearcherAgent."""

import httpx

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

_TAVILY_ENDPOINT = "https://api.tavily.com/search"


class SearchClient:
    """Search client backed by Tavily, with a deterministic offline mock fallback.

    When `TAVILY_API_KEY` is not configured, `search` falls back to a local mock so the
    workflow stays runnable offline (see docs/lab_guide.md troubleshooting notes).
    """

    def __init__(self, api_key: str | None = None, timeout: float | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.tavily_api_key
        self.timeout = timeout if timeout is not None else float(settings.timeout_seconds)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if not self.api_key:
            return self._mock_search(query, max_results)
        return self._tavily_search(query, max_results)

    def _tavily_search(self, query: str, max_results: int) -> list[SourceDocument]:
        try:
            response = httpx.post(
                _TAVILY_ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise AgentExecutionError(f"Tavily search failed: {exc}") from exc

        results = payload.get("results", [])[:max_results]
        sources = [
            SourceDocument(
                title=result.get("title") or query,
                url=result.get("url"),
                snippet=(result.get("content") or "").strip()[:600],
                metadata={"score": result.get("score"), "provider": "tavily"},
            )
            for result in results
        ]
        if not sources:
            raise AgentExecutionError(f"Tavily returned no results for query: {query!r}")
        return sources

    def _mock_search(self, query: str, max_results: int) -> list[SourceDocument]:
        return [
            SourceDocument(
                title=f"Mock source {i + 1} for '{query}'",
                url=None,
                snippet=(
                    f"Offline placeholder snippet {i + 1}. Set TAVILY_API_KEY in .env to "
                    "replace this with real web search results."
                ),
                metadata={"provider": "mock"},
            )
            for i in range(max_results)
        ]
