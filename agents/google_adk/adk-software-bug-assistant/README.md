# adk-software-bug-assistant

Source: https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/software-bug-assistant

Bug triage agent over the support-ticket database with KB lookup, a web-search sub-agent tool and Stack Exchange search.

## Shape

single LlmAgent with ticket/KB/HR tools, AgentTool(search_agent) and a Stack Exchange search tool

## Adaptations

- MCP Toolbox ticket tools (Postgres) replaced by `tickets_search` / `tickets_get` / `tickets_update`, plus `kb_search`, `kb_get_article`, `hr_search_employees`.
- Google Search sub-agent uses `web_search` + `fetch_url`; LangChain StackExchangeTool becomes `web_search` restricted to Stack Exchange sites; the GitHub MCP toolset (needs a token) is dropped.
- Company changed from QuantumRoast to Lumen & Co.

## Model and tracing

- Complexity: simple. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

tickets_search, tickets_get, tickets_update, kb_search, kb_get_article, hr_search_employees, web_search, fetch_url

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 3.0 LLM generations per run.
Mean cost $0.0022 per run (Langfuse trace cost).
