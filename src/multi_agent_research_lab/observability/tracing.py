"""Tracing hooks.

`trace_span` always records a local duration span (used for `ResearchState.trace`), and
additionally mirrors the span to LangSmith when `LANGSMITH_API_KEY` is configured, so the
same call sites work with `make run-multi` offline or with a trace UI open. Nested spans
(e.g. per-agent spans opened while `MultiAgentWorkflow.run` holds an outer span) nest
under one trace automatically via `tracing_context`, instead of showing up as unrelated
top-level traces. Provider errors (e.g. no network) are swallowed so tracing can never
break an agent run.
"""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

_langsmith_trace: Any = None
_langsmith_tracing_context: Any = None
with suppress(ImportError):  # langsmith is an optional `llm` extra
    from langsmith.run_helpers import trace as _langsmith_trace
    from langsmith.run_helpers import tracing_context as _langsmith_tracing_context

_client_cache: dict[str, Any] = {}


def _get_langsmith_client(api_key: str) -> Any:
    client = _client_cache.get(api_key)
    if client is None:
        from langsmith import Client

        client = Client(api_key=api_key)
        _client_cache[api_key] = client
    return client


def flush_traces() -> None:
    """Force-deliver any pending LangSmith spans.

    LangSmith batches spans on a background thread; call this before a short-lived
    process (e.g. a CLI command) exits so the trace is actually visible in the UI.
    """

    settings = get_settings()
    if settings.langsmith_api_key:
        with suppress(Exception):  # noqa: BLE001 - tracing must never break the caller
            _get_langsmith_client(settings.langsmith_api_key).flush()


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Open a span, yielding a mutable dict callers can enrich before it closes."""

    settings = get_settings()
    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    stack: ExitStack | None = None
    run_tree = None
    if (
        _langsmith_trace is not None
        and _langsmith_tracing_context is not None
        and settings.langsmith_api_key
    ):
        try:
            client = _get_langsmith_client(settings.langsmith_api_key)
            stack = ExitStack()
            stack.enter_context(
                _langsmith_tracing_context(
                    enabled=True, client=client, project_name=settings.langsmith_project
                )
            )
            run_tree = stack.enter_context(
                _langsmith_trace(
                    name=name,
                    run_type="chain",
                    inputs=attributes or {},
                    project_name=settings.langsmith_project,
                    client=client,
                )
            )
        except Exception:  # noqa: BLE001 - tracing must never break the caller
            if stack is not None:
                with suppress(Exception):
                    stack.close()
            stack, run_tree = None, None

    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
        if stack is not None:
            if run_tree is not None:
                with suppress(Exception):
                    run_tree.end(outputs={"duration_seconds": span["duration_seconds"]})
            with suppress(Exception):
                stack.close()
