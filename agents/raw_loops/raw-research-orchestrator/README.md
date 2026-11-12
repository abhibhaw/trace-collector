# raw-research-orchestrator

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/orchestrator_workers.ipynb

The cookbook's FlexibleOrchestrator (orchestrator returns <analysis> + <tasks>, workers process each task), with prompts from patterns/agents/prompts/research_lead_agent.md and research_subagent.md.

## What was adapted

- Workers are tool-using research loops (web, URL fetch, Wikipedia, arXiv; max 4 turns) and run concurrently with asyncio.
- Added a synthesis step in which the lead writes a cited report.
- Lead (plan + synthesis) runs on claude-sonnet-5-5; workers run on claude-haiku-5-5. AGENT.model names the lead model.

## Tools

web_search, fetch_url, wikipedia_search, wikipedia_page, arxiv_search

## Trace shape

agent > orchestrator_plan > sonnet generation; workers > 2-4 parallel worker agents > turns > haiku generations + tools; synthesize > sonnet generation

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 14.0. Mean cost per run: USD 0.0404 (Langfuse calculated cost). Sonnet lead plus 3-4 Haiku workers; arXiv sometimes returns HTTP 429 under parallel load (recorded as tool warnings).
