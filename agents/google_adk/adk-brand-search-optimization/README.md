# adk-brand-search-optimization

Source: https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/contrib/python/brand-search-optimization

Coordinator hands off via transfer_to_agent from catalog keyword mining to competitor search to a generator/critic title-comparison team.

## Shape

root LlmAgent -> transfer_to_agent chain keyword_finding_agent -> search_results_agent -> comparison_root_agent[comparison_generator_agent, comparison_critic_agent]

## Adaptations

- The BigQuery catalog tool becomes `get_product_details_for_brand` over `products_search` (brand Lumen & Co).
- Gemini Computer Use browsing of Google Shopping is replaced by `web_search` / `fetch_url`.
- A simulated brand manager answers the brand question and asks to continue.

## Model and tracing

- Complexity: medium. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

products_search, inventory_check, web_search, fetch_url

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 12.0 LLM generations per run.
Mean cost $0.0161 per run (Langfuse trace cost).
