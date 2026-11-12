# adk-financial-advisor

Source: https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/contrib/python/financial-advisor

Financial coordinator that runs data, trading, execution and risk analyst sub-agents as tools, with a simulated investor answering its questions.

## Shape

financial_coordinator LlmAgent with AgentTool(data_analyst, trading_analyst, execution_analyst, risk_analyst); output_key state hand-off; multi-turn simulated investor

## Adaptations

- The data analyst uses `finance_quote`, `finance_history` (synthetic prices) and `web_search` instead of Google Search.
- The coordinator asks for ticker, risk attitude, horizon and execution preferences; a simulated investor answers (up to 4 turns).

## Model and tracing

- Complexity: medium. Model: claude-haiku-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

finance_quote, finance_history, web_search

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 12.3 LLM generations per run.
Mean cost $0.0138 per run (Langfuse trace cost).
