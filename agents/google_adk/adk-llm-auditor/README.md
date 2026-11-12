# adk-llm-auditor

Source: https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/contrib/python/llm-auditor

SequentialAgent critic->reviser that verifies claims in an LLM answer with web search and minimally rewrites it.

## Shape

SequentialAgent[critic_agent (web_search, fetch_url; after_tool + after_model callbacks), reviser_agent (after_model callback)]

## Adaptations

- Google Search grounding replaced by `web_search` and `fetch_url`.
- The grounding-reference callback (`_render_reference`) now renders references that an `after_tool_callback` collects from search results into session state.
- The reviser's end-of-edit callback got a `None` guard on `part.text`.

## Model and tracing

- Complexity: simple. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

web_search, fetch_url

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 3.3 LLM generations per run.
Mean cost $0.0019 per run (Langfuse trace cost).
