# adk-academic-research

Source: https://github.com/google/adk-samples/tree/a9dc17718dc8aa2388a0c170263ef05d6e964147/python/agents/academic-research

Coordinator analyzes a seminal paper, then calls a citing-paper search agent and a research-directions agent as tools.

## Shape

LlmAgent coordinator with arxiv_search/fetch_url and AgentTool(academic_websearch_agent), AgentTool(academic_newresearch_agent); output_key state templating

## Adaptations

- The seminal paper is looked up with `arxiv_search` or `fetch_url` instead of a PDF upload.
- Google Search in the citing-paper agent is replaced by `arxiv_search` + `web_search`; the target count drops from 10 to 5 papers per year to bound cost.
- Initial session state seeds `seminal_paper`, `recent_citing_papers`, `current_date` for the instruction templates.

## Model and tracing

- Complexity: simple. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

arxiv_search, web_search, fetch_url

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 7.0 LLM generations per run.
Mean cost $0.0093 per run (Langfuse trace cost).
