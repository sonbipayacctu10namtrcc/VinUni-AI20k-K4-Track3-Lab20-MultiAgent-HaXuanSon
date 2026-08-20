"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(
    metrics: list[BenchmarkMetrics], *, title: str = "Benchmark Report"
) -> str:
    """Render benchmark metrics to markdown: a results table plus a failure-mode section."""

    lines = [
        f"# {title}",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        citation = "n/a" if item.citation_coverage is None else f"{item.citation_coverage:.0%}"
        failure = "" if item.failure_rate is None else f"{item.failure_rate:.0%}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} "
            f"| {citation} | {failure} | {item.notes} |"
        )

    lines += ["", "## Failure modes", ""]
    failures = [item for item in metrics if (item.failure_rate or 0) > 0]
    if not failures:
        lines.append("No failed runs recorded.")
    else:
        for item in failures:
            reason = item.notes or "no notes recorded"
            lines.append(f"- **{item.run_name}**: {reason}")

    lines += ["", "## Summary", "", _summarize(metrics)]

    return "\n".join(lines) + "\n"


def _summarize(metrics: list[BenchmarkMetrics]) -> str:
    """One paragraph comparing baseline vs. multi-agent averages, when both are present."""

    by_run: dict[str, list[BenchmarkMetrics]] = {}
    for item in metrics:
        by_run.setdefault(item.run_name, []).append(item)

    def avg(items: list[BenchmarkMetrics], field: str) -> float | None:
        values = [getattr(item, field) for item in items if getattr(item, field) is not None]
        return sum(values) / len(values) if values else None

    parts = []
    for run_name, items in by_run.items():
        latency = avg(items, "latency_seconds")
        quality = avg(items, "quality_score")
        coverage = avg(items, "citation_coverage")
        failure = avg(items, "failure_rate")
        parts.append(
            f"**{run_name}** (n={len(items)}): avg latency "
            f"{latency:.2f}s" if latency is not None else f"**{run_name}** (n={len(items)})"
        )
        if quality is not None:
            parts[-1] += f", avg quality {quality:.1f}/10"
        if coverage is not None:
            parts[-1] += f", avg citation coverage {coverage:.0%}"
        if failure is not None:
            parts[-1] += f", failure rate {failure:.0%}"
        parts[-1] += "."

    return " ".join(parts) if parts else "No metrics recorded."
