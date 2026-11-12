# raw-parallel-stock-brief

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/basic_workflows.ipynb

The cookbook's `parallel()` workflow, in both of its forms from "Building effective agents": sectioning and voting.

## What was adapted

- ThreadPoolExecutor replaced by asyncio.gather so spans keep their parent.
- Three tool-using workers (market data, sector context, news and risks) run concurrently; an aggregator writes the brief; three reviewers vote in parallel; a majority REVISE triggers one rewrite.

## Tools

finance_quote, finance_history, wikipedia_search, wikipedia_page, web_search

## Trace shape

agent > scope; sectioning > 3 parallel worker agents > turns > tools; aggregate; voting (guardrail) > 3 reviewer evaluators; optional revise

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 12.7. Mean cost per run: USD 0.0044 (Langfuse calculated cost).
