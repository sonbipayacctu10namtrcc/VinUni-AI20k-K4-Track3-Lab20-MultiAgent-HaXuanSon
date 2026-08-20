"""Benchmark harness for single-agent vs multi-agent runs."""

import re
from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState | None, BenchmarkMetrics]:
    """Run `runner(query)`, measuring latency and deriving cost/quality/citation/failure
    metrics from the resulting state. Never raises: a runner exception is itself a
    benchmark result (failure_rate=1.0) so a bad query doesn't abort the whole sweep.
    """

    started = perf_counter()
    try:
        state = runner(query)
    except Exception as exc:  # noqa: BLE001 - a runner failure is a benchmark result
        latency = perf_counter() - started
        return None, BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"runner raised {type(exc).__name__}: {exc}",
        )
    latency = perf_counter() - started

    citation_coverage = _citation_coverage(state)
    failed = not state.final_answer or bool(state.errors)
    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=_estimate_cost(state),
        quality_score=_quality_score(state, citation_coverage),
        citation_coverage=citation_coverage,
        failure_rate=1.0 if failed else 0.0,
        notes="; ".join(state.errors),
    )
    return state, metrics


def _estimate_cost(state: ResearchState) -> float | None:
    costs: list[float] = [
        cost
        for result in state.agent_results
        if (cost := result.metadata.get("cost_usd")) is not None
    ]
    return sum(costs) if costs else None


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of `state.sources` referenced by a `[n]` marker in the final answer.

    None (not 0.0) when there were no sources to cite, e.g. the single-agent baseline,
    so it doesn't get unfairly compared against a multi-agent run that could cite.
    """

    if not state.sources:
        return None
    cited = {int(match) for match in _CITATION_PATTERN.findall(state.final_answer or "")}
    valid = {n for n in cited if 1 <= n <= len(state.sources)}
    return len(valid) / len(state.sources)


def _quality_score(state: ResearchState, citation_coverage: float | None) -> float | None:
    """Cheap automated proxy (0-10): substance + citation coverage - errors.

    This is not a substitute for the human rubric in docs/peer_review_rubric.md; it only
    exists so the benchmark table has a number to sort/compare runs by.
    """

    if not state.final_answer:
        return 0.0
    word_count = len(state.final_answer.split())
    length_score = min(word_count / 150, 1.0) * 4
    coverage_score = (citation_coverage or 0.0) * 4
    error_penalty = 2.0 if state.errors else 0.0
    return round(max(0.0, min(10.0, 2.0 + length_score + coverage_score - error_penalty)), 1)
