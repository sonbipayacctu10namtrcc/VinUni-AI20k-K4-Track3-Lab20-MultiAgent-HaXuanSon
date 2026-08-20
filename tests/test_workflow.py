"""End-to-end test of the compiled LangGraph, using fake clients (no network calls)."""

from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from tests.test_workers import FakeLLMClient, FakeSearchClient


def test_workflow_runs_full_pipeline_and_stops() -> None:
    sources = [SourceDocument(title="Doc A", url="https://a", snippet="about A")]
    workflow = MultiAgentWorkflow(
        llm_client=FakeLLMClient("Summary citing [1]."),
        search_client=FakeSearchClient(sources),
    )
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = workflow.run(state)

    assert result.route_history == [
        "researcher",
        "analyst",
        "writer",
        "critic",
        "done",
    ]
    assert result.final_answer is not None
    assert "Summary citing [1]." in result.final_answer
    assert result.critic_notes is not None
    assert result.errors == []
