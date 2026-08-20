"""Unit tests for SupervisorAgent's routing policy.

Replaces the old skeleton guard test (test_agents_todo.py) now that the TODO is
implemented, per the NOTE(student) that used to live there.
"""

from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_routes_to_researcher_when_no_sources() -> None:
    state = SupervisorAgent().run(_state())
    assert state.route_history == ["researcher"]
    assert state.iteration == 1


def test_routes_to_analyst_once_research_notes_exist() -> None:
    state = _state()
    state.sources = [SourceDocument(title="A", snippet="...")]
    state.research_notes = "[1] A: ..."
    state = SupervisorAgent().run(state)
    assert state.route_history == ["analyst"]


def test_routes_to_writer_once_analysis_notes_exist() -> None:
    state = _state()
    state.sources = [SourceDocument(title="A", snippet="...")]
    state.research_notes = "[1] A: ..."
    state.analysis_notes = "Key claim: ..."
    state = SupervisorAgent().run(state)
    assert state.route_history == ["writer"]


def test_routes_to_critic_once_final_answer_exists() -> None:
    state = _state()
    state.sources = [SourceDocument(title="A", snippet="...")]
    state.research_notes = "[1] A: ..."
    state.analysis_notes = "Key claim: ..."
    state.final_answer = "Answer [1]."
    state = SupervisorAgent().run(state)
    assert state.route_history == ["critic"]


def test_routes_to_done_once_pipeline_is_complete() -> None:
    state = _state()
    state.sources = [SourceDocument(title="A", snippet="...")]
    state.research_notes = "[1] A: ..."
    state.analysis_notes = "Key claim: ..."
    state.final_answer = "Answer [1]."
    state.critic_notes = "Citation coverage: 100% (1/1 sources cited)."
    state = SupervisorAgent().run(state)
    assert state.route_history == ["done"]


def test_stops_at_max_iterations_even_if_incomplete() -> None:
    state = _state()
    supervisor = SupervisorAgent(max_iterations=2)
    state = supervisor.run(state)  # iteration -> 1, still incomplete -> researcher
    state = supervisor.run(state)  # iteration -> 2, still incomplete -> researcher
    state = supervisor.run(state)  # iteration(2) >= max_iterations(2) -> done
    assert state.route_history == ["researcher", "researcher", "done"]
    assert any("max_iterations" in error for error in state.errors)
