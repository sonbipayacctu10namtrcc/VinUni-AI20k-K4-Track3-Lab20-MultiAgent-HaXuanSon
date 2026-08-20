"""Command-line entrypoint for the lab starter."""

from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError, StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import flush_traces
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore
from multi_agent_research_lab.utils.timer import elapsed_timer

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()

_BASELINE_SYSTEM_PROMPT = (
    "You are a single-agent research assistant with no external tools. Answer the "
    "user's research query directly and concisely from your own knowledge, and say so "
    "plainly when you are not certain rather than inventing sources."
)


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


def _run_baseline_state(query: str) -> ResearchState:
    """Single-agent baseline: one direct LLM call, no tools, no routing."""

    request = _parse_query(query)
    state = ResearchState(request=request)
    client = LLMClient()
    with elapsed_timer() as elapsed:
        response = client.complete(_BASELINE_SYSTEM_PROMPT, request.query)
    latency = elapsed()
    state.final_answer = response.content
    state.agent_results.append(
        AgentResult(
            agent=AgentName.WRITER,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
                "latency_seconds": latency,
            },
        )
    )
    state.add_trace_event(
        "baseline.completed",
        {
            "latency_seconds": latency,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        },
    )
    return state


def _run_multi_agent_state(query: str) -> ResearchState:
    request = _parse_query(query)
    state = ResearchState(request=request)
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the single-agent baseline: one direct LLM call, no tools, no routing."""

    _init()
    try:
        state = _run_baseline_state(query)
    except AgentExecutionError as exc:
        console.print(Panel.fit(str(exc), title="Baseline Error", style="red"))
        raise typer.Exit(code=1) from exc

    console.print(Panel.fit(state.final_answer or "", title="Single-Agent Baseline"))
    usage = state.agent_results[-1].metadata
    console.print(
        f"latency={usage['latency_seconds']:.2f}s "
        f"input_tokens={usage['input_tokens']} output_tokens={usage['output_tokens']} "
        f"cost_usd={usage['cost_usd']}"
    )


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow: supervisor routes researcher/analyst/writer/critic."""

    _init()
    try:
        result = _run_multi_agent_state(query)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    finally:
        flush_traces()
    console.print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    config_path: Annotated[
        str, typer.Option("--config", "-c", help="Path to a lab config YAML")
    ] = "configs/lab_default.yaml",
    output: Annotated[
        str, typer.Option("--output", "-o", help="Report path, relative to reports/")
    ] = "benchmark_report.md",
) -> None:
    """Run baseline and multi-agent over the configured queries and write a report."""

    _init()
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    queries: list[str] = config["benchmark"]["queries"]

    metrics = []
    for query in queries:
        console.print(f"[bold]Running baseline[/bold]: {query}")
        _, baseline_metrics = run_benchmark("baseline", query, _run_baseline_state)
        metrics.append(baseline_metrics)

        console.print(f"[bold]Running multi-agent[/bold]: {query}")
        _, multi_metrics = run_benchmark("multi-agent", query, _run_multi_agent_state)
        metrics.append(multi_metrics)

    flush_traces()
    report = render_markdown_report(metrics)
    path = LocalArtifactStore().write_text(output, report)
    console.print(Panel.fit(f"Report written to {path}", title="Benchmark"))
    console.print(report)


if __name__ == "__main__":
    app()
