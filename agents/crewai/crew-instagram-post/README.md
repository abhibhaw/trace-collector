# crew-instagram-post

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/instagram_post

Review the photos you got from the senior photographer. Make sure they are the best possible and 

## What was adapted

Instagram post crews, adapted from crewAI-examples/crews/instagram_post.

Shape kept from the source: two crews run back to back. The copy crew (lead market analyst, chief marketing
strategist, creative content creator) does product analysis, competitor analysis, campaign development and
ad copy. The image crew (senior photographer, chief creative director with delegation) turns the copy into
three photo briefs and reviews them, delegating follow-up work when needed.
Adapted: browser/search tools are replaced by the product catalog, inventory, knowledge base, and web search.
The "product website" is a Lumen & Co product (or category) named in the request.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: medium.
- Shared tools: fetch_url, inventory_check, kb_search, products_search, web_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 12.0 (per task: 10, 12, 14).
Mean cost per run: $0.0222 (Langfuse trace cost).
