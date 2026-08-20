"""Critic agent: a lightweight, deterministic quality gate.

No LLM call here on purpose — citation coverage and hallucinated citation indices are
checkable with plain string parsing, which is cheaper and more reliable than asking a
model to grade itself.
"""

import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class CriticAgent(BaseAgent):
    """Validates the final answer's citation coverage before the run is marked done."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("critic.check", {"query": state.request.query}) as span:
            answer = state.final_answer or ""
            total_sources = len(state.sources)
            cited = {int(match) for match in _CITATION_PATTERN.findall(answer)}
            valid_cited = {n for n in cited if 1 <= n <= total_sources}
            invalid_cited = cited - valid_cited
            coverage = (len(valid_cited) / total_sources) if total_sources else 0.0
            span["attributes"]["citation_coverage"] = coverage

        findings = [
            f"Citation coverage: {coverage:.0%} ({len(valid_cited)}/{total_sources} "
            "sources cited)."
        ]
        if total_sources and not valid_cited:
            findings.append("Warning: final answer cites no known source.")
            state.errors.append("critic: final answer has no valid citations")
        if invalid_cited:
            findings.append(
                f"Warning: citation indices {sorted(invalid_cited)} do not match any source."
            )
            state.errors.append("critic: invalid citation indices found")

        state.critic_notes = "\n".join(findings)
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=state.critic_notes,
                metadata={"citation_coverage": coverage},
            )
        )
        state.add_trace_event("critic.completed", {"citation_coverage": coverage})
        return state
