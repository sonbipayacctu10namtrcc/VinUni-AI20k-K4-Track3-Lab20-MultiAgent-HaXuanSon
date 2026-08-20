"""Writer agent."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM_PROMPT = (
    "You are a technical writer. Synthesize the research notes and analysis into a clear, "
    "well-organized answer for the described audience. Cite sources inline using the same "
    "[n] numbering as the research notes; never invent a source or a number that isn't in "
    "the notes. If the notes or analysis are missing or say no evidence was found, say so "
    "plainly instead of guessing."
)


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        user_prompt = (
            f"Audience: {state.request.audience}\n"
            f"Research question: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or '(no notes available)'}\n\n"
            f"Analysis:\n{state.analysis_notes or '(no analysis available)'}"
        )
        with trace_span("writer.write", {"query": state.request.query}) as span:
            response = self.llm_client.complete(_SYSTEM_PROMPT, user_prompt, temperature=0.4)

        answer = response.content.strip()
        if state.sources:
            references = "\n".join(
                f"[{i + 1}] {source.title}" + (f" — {source.url}" if source.url else "")
                for i, source in enumerate(state.sources)
            )
            answer = f"{answer}\n\nSources:\n{references}"

        state.final_answer = answer
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=answer,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            )
        )
        state.add_trace_event("writer.completed", {"duration_seconds": span["duration_seconds"]})
        return state
