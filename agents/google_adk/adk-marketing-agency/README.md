# adk-marketing-agency

Source: https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/marketing-agency

Marketing coordinator that calls domain, website, campaign and logo sub-agents as tools; logos are saved as SVG artifacts.

## Shape

marketing_coordinator with AgentTool(domain_create, website_create, marketing_create, logo_create); logo saved as an ADK artifact; simulated client turns

## Adaptations

- Domain availability: Google Search replaced by a local DNS-lookup tool `check_domains` plus `web_search`.
- Imagen replaced by a local `generate_image` tool that renders an SVG logo and stores it with `tool_context.save_artifact`; `load_artifacts` kept.
- A simulated client supplies keywords and picks the first suggested domain.

## Model and tracing

- Complexity: medium. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

web_search + local check_domains (DNS), generate_image (SVG artifact), load_artifacts

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 11.3 LLM generations per run.
Mean cost $0.0079 per run (Langfuse trace cost).
