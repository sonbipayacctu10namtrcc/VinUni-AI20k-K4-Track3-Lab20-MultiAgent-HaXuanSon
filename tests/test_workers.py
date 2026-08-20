"""Unit tests for the worker agents, using fake LLM/search clients (no network calls)."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient, LLMResponse
from multi_agent_research_lab.services.search_client import SearchClient


class FakeLLMClient(LLMClient):
    def __init__(self, content: str) -> None:  # intentionally skip LLMClient.__init__
        self.content = content
        self.calls: list[tuple[str, str, float]] = []

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2):
        self.calls.append((system_prompt, user_prompt, temperature))
        return LLMResponse(content=self.content, input_tokens=10, output_tokens=20, cost_usd=0.001)


class FakeSearchClient(SearchClient):
    def __init__(self, sources: list[SourceDocument]) -> None:  # skip SearchClient.__init__
        self.sources = sources

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return self.sources[:max_results]


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_researcher_populates_sources_and_notes() -> None:
    sources = [SourceDocument(title="Doc A", url="https://a", snippet="about A")]
    state = ResearcherAgent(FakeSearchClient(sources)).run(_state())
    assert state.sources == sources
    assert "[1] Doc A" in state.research_notes
    assert state.agent_results[-1].metadata["source_count"] == 1


def test_analyst_populates_analysis_notes() -> None:
    state = _state()
    state.research_notes = "[1] Doc A: about A"
    fake = FakeLLMClient("Claim 1 is supported by [1].")
    state = AnalystAgent(fake).run(state)
    assert state.analysis_notes == "Claim 1 is supported by [1]."
    assert fake.calls[0][2] == 0.1  # analyst uses a low temperature


def test_writer_appends_sources_and_cites() -> None:
    state = _state()
    state.sources = [SourceDocument(title="Doc A", url="https://a", snippet="about A")]
    state.research_notes = "[1] Doc A: about A"
    state.analysis_notes = "Claim 1 is supported by [1]."
    fake = FakeLLMClient("Multi-agent systems are useful [1].")
    state = WriterAgent(fake).run(state)
    assert "Multi-agent systems are useful [1]." in state.final_answer
    assert "Sources:" in state.final_answer
    assert "https://a" in state.final_answer


def test_critic_flags_missing_citations() -> None:
    state = _state()
    state.sources = [SourceDocument(title="Doc A", snippet="about A")]
    state.final_answer = "An answer with no citation at all."
    state = CriticAgent().run(state)
    assert "0%" in state.critic_notes
    assert any("no valid citations" in error for error in state.errors)


def test_critic_accepts_valid_citation() -> None:
    state = _state()
    state.sources = [SourceDocument(title="Doc A", snippet="about A")]
    state.final_answer = "An answer citing [1]."
    state = CriticAgent().run(state)
    assert "100%" in state.critic_notes
    assert state.errors == []
