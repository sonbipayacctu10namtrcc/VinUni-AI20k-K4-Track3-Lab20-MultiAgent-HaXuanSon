"""Researcher agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise, citable research notes.

    Raises `AgentExecutionError` on search failure instead of swallowing it: the
    workflow's node wrapper (see `graph/workflow.py`) is the single place that decides
    how to react to a failed worker (record + let the supervisor retry or give up).
    """

    name = "researcher"

    def __init__(self, search_client: SearchClient | None = None) -> None:
        self.search_client = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("researcher.search", {"query": state.request.query}) as span:
            sources = self.search_client.search(
                state.request.query, max_results=state.request.max_sources
            )
            span["attributes"]["source_count"] = len(sources)

        state.sources = sources
        state.research_notes = "\n".join(
            f"[{i + 1}] {source.title}: {source.snippet}" for i, source in enumerate(sources)
        )

        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=state.research_notes,
                metadata={"source_count": len(sources)},
            )
        )
        state.add_trace_event(
            "researcher.completed",
            {"source_count": len(sources), "duration_seconds": span["duration_seconds"]},
        )
        return state
