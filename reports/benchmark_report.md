# Benchmark Report

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 9.62 | 0.0004 | 6.0 | n/a | 0% |  |
| multi-agent | 15.58 | 0.0009 | 10.0 | 100% | 0% |  |
| baseline | 5.31 | 0.0002 | 6.0 | n/a | 0% |  |
| multi-agent | 15.71 | 0.0009 | 10.0 | 100% | 0% |  |
| baseline | 4.28 | 0.0002 | 6.0 | n/a | 0% |  |
| multi-agent | 12.98 | 0.0009 | 10.0 | 100% | 0% |  |

## Failure modes

No failed runs recorded.

## Summary

**baseline** (n=3): avg latency 6.40s, avg quality 6.0/10, failure rate 0%. **multi-agent** (n=3): avg latency 14.76s, avg quality 10.0/10, avg citation coverage 100%, failure rate 0%.

## Failure mode analysis

This run of the 3 configured queries had a 0% failure rate on both paths, but the
`quality_score` gap (6.0 vs 10.0) is mostly a citation-coverage artifact: the heuristic
scorer in `evaluation/benchmark.py` gives up to 4/10 points for citing sources, and the
single-agent baseline has no tool access, so it *cannot* cite anything — its `n/a`
coverage isn't a defect, it's a structural limitation of the architecture being tested.
Judged only on prose quality, the two are closer than the score implies.

The multi-agent path is ~2.3x slower (14.8s vs 6.4s avg) because it is sequential:
search, then 2 additional LLM calls (analyst, writer) plus a citation check, versus the
baseline's single call. None of that latency is hidden or parallelized — the supervisor
loop revisits one worker per hop, so cost and latency scale linearly with pipeline depth.

Failure modes that exist by design but weren't triggered in this run:

- **Search failure or empty results** (`services/search_client.py`): a Tavily error or
  a query with genuinely no results raises `AgentExecutionError` inside the graph node,
  which `graph/workflow.py`'s `_guarded` wrapper turns into a `state.errors` entry
  instead of a crash. `research_notes` stays empty, so the supervisor retries the
  researcher on the next hop — up to `max_iterations` (default 6) before giving up with
  `final_answer` still unset. This is the "retry" branch of the guardrail question in
  `docs/lab_guide.md`.
- **LLM provider failure** (`services/llm_client.py`): `LLMClient._call` retries
  transient `OpenAIError`s up to 3 times with exponential backoff before raising; the
  same `_guarded` wrapper then records it and the supervisor retries that worker.
- **Hallucinated or missing citations** (`agents/critic.py`): if the writer cites a
  source index that doesn't exist, or writes a final answer with none of the notes'
  `[n]` markers, the critic appends a low-coverage or invalid-citation entry to
  `state.errors` — visible in the benchmark's Notes column and in `route_history`, since
  the run still completes (citation problems are a quality signal, not a hard stop).
- **Runaway iteration**: if a worker keeps failing, `SupervisorAgent` forces `done` once
  `state.iteration >= max_iterations` and logs why, so a bad query degrades to a
  (possibly incomplete) answer plus an error trail instead of an infinite loop.

None of these paths were exercised by the 3 benchmark queries above, which is itself a
gap in this benchmark set — see the exit ticket in `docs/lab_guide.md` for when
multi-agent's extra latency/cost is (and isn't) worth it.
