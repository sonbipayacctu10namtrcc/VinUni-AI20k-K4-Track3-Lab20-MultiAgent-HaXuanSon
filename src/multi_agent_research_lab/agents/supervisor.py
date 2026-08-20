"""Supervisor / router."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState

DONE = "done"
ROUTES = ("researcher", "analyst", "writer", "critic", DONE)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing is a simple field-presence policy: run the next worker whose required
    output is still missing, in a fixed pipeline order. This keeps the policy legible
    and testable, and naturally retries a worker that failed to fill its field (e.g. a
    flaky search call) until `max_iterations` is hit.
    """

    name = "supervisor"

    def __init__(self, max_iterations: int | None = None) -> None:
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            self.max_iterations = get_settings().max_iterations

    def run(self, state: ResearchState) -> ResearchState:
        next_route = self._decide(state)
        state.record_route(next_route)
        state.add_trace_event(
            "supervisor.route", {"next": next_route, "iteration": state.iteration}
        )
        return state

    def _decide(self, state: ResearchState) -> str:
        if state.iteration >= self.max_iterations:
            if not state.final_answer:
                state.errors.append(
                    f"supervisor: max_iterations={self.max_iterations} reached before a "
                    "final_answer was produced"
                )
            return DONE
        if not state.sources or not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        if state.critic_notes is None:
            return "critic"
        return DONE
