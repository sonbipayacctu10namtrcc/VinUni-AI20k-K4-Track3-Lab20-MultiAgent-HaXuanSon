"""LangGraph workflow.

Keep orchestration here; keep agent internals in `agents/`.
"""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import DONE, SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph: supervisor routes to one worker at a
    time, each worker hands back to supervisor, and supervisor stops the run once the
    pipeline is complete or `max_iterations` is reached (see `SupervisorAgent`).
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        llm_client = llm_client or LLMClient()
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent(search_client or SearchClient())
        self.analyst = AnalystAgent(llm_client)
        self.writer = WriterAgent(llm_client)
        self.critic = CriticAgent()
        self._graph: Any = self.build()

    def build(self) -> Any:
        """Create and compile the LangGraph graph.

        Nodes: supervisor, researcher, analyst, writer, critic. The supervisor is the
        only node with conditional edges; every worker routes straight back to it.
        """

        graph = StateGraph(ResearchState)
        graph.add_node("supervisor", self.supervisor.run)
        graph.add_node("researcher", self._guarded(self.researcher))  # type: ignore[arg-type]
        graph.add_node("analyst", self._guarded(self.analyst))  # type: ignore[arg-type]
        graph.add_node("writer", self._guarded(self.writer))  # type: ignore[arg-type]
        graph.add_node("critic", self._guarded(self.critic))  # type: ignore[arg-type]

        graph.set_entry_point("supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state.route_history[-1],
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                DONE: END,
            },
        )
        for node in ("researcher", "analyst", "writer", "critic"):
            graph.add_edge(node, "supervisor")

        return graph.compile()

    @staticmethod
    def _guarded(agent: BaseAgent) -> Callable[[ResearchState], ResearchState]:
        """Wrap a worker so a failed call is recorded instead of crashing the run.

        The state is returned unchanged on failure: the field the worker was supposed
        to fill stays empty, so the supervisor will route back to the same worker
        (retry) until `max_iterations` forces a stop.
        """

        def _node(state: ResearchState) -> ResearchState:
            try:
                return agent.run(state)
            except AgentExecutionError as exc:
                state.errors.append(f"{agent.name}: {exc}")
                return state

        return _node

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return the final state.

        The whole invoke() call is wrapped in one outer span so that, when LangSmith is
        configured, every per-agent span opened inside a node (see e.g.
        `researcher.search` in `agents/researcher.py`) nests under a single trace for
        this run instead of showing up as unrelated top-level traces.
        """

        settings = get_settings()
        with trace_span("multi_agent_workflow.run", {"query": state.request.query}):
            raw: dict[str, Any] = self._graph.invoke(
                state, config={"recursion_limit": settings.max_iterations * 3 + 5}
            )
        return ResearchState.model_validate(raw)
