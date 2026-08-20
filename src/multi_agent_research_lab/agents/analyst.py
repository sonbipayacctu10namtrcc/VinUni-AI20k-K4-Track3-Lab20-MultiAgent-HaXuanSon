"""Analyst agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a careful research analyst. You are given numbered research notes "
    "(format '[n] title: snippet'). Extract the key claims, compare viewpoints across "
    "sources, note any contradictions, and flag claims with weak or missing evidence. "
    "Refer to sources using the same [n] numbering as the notes. Be concise: use short "
    "bullet points, not prose paragraphs."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        user_prompt = (
            f"Research question: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or '(no notes available)'}"
        )
        with trace_span("analyst.analyze", {"query": state.request.query}) as span:
            response = self.llm_client.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.1)

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("analyst.completed", {"duration_seconds": span["duration_seconds"]})
        return state
