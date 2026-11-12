# crew-marketing-strategy

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/marketing_strategy

Develop 2-3 creative marketing campaign ideas aligned with the strategy, then create one marketing 

## What was adapted

Marketing strategy crew, adapted from crewAI-examples/crews/marketing_strategy (marketing_posts).

Shape kept from the source: lead market analyst, chief marketing strategist, and creative content creator with the
source's pydantic models (MarketStrategy, CampaignIdea, Copy) as structured task outputs. The source ships a
commented-out `Process.hierarchical` option and an unused `chief_creative_director` agent; this adaptation turns
both on: the chief creative director (Sonnet) is the manager agent that plans, delegates every task to a coworker,
and reviews the result.
Adapted: Serper/ScrapeWebsite are replaced with web search, URL fetch, product catalog and knowledge base.
To bound cost the five source tasks are merged into three (research, strategy, campaign+copy).

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-sonnet-5-5 for the manager (chief creative director), claude-haiku-5-5 for the three specialists.
- Complexity: complex.
- Shared tools: fetch_url, kb_search, products_search, web_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 10.0 (per task: 10, 10, 10).
Mean cost per run: $0.0874 (Langfuse trace cost).
Sonnet manager calls are most of the cost.
