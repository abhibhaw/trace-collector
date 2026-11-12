# adk-deep-search

Source: https://github.com/google/adk-samples/tree/b4fa327b91a1a2c3d710f7476ea723f42d1c717c/core/python/deep-search

Interactive research planner that, once the plan is approved, runs a sequential pipeline with an evaluate-and-refine LoopAgent and a cited report composer.

## Shape

interactive_planner (AgentTool plan_generator) -> transfer -> SequentialAgent[section_planner, section_researcher, LoopAgent(max 2)[research_evaluator, EscalationChecker (custom BaseAgent), enhanced_search_executor], report_composer]; two user turns (plan, then simulated approval)

## Adaptations

- Google Search grounding replaced by `web_search` / `fetch_url`; the source-collection callback is now an `after_tool_callback` that gives each URL a `src-N` id, so `citation_replacement_callback` still links citations.
- Workers use claude-haiku-5-5; the evaluator and report composer use claude-sonnet-5-5 (source worker_model / critic_model split).
- Sonnet rejects the forced `tool_choice` that ADK uses for `output_schema` on Claude, so the evaluator emits JSON text; an `after_agent_callback` validates it against the `Feedback` model.
- `BuiltInPlanner(thinking_config)` removed (the proxy forbids thinking budgets). LoopAgent max_iterations=2 and search caps keep a run within about 25 LLM calls.
- Human plan approval is simulated by a second user turn: "Looks good, run it."

## Model and tracing

- Complexity: complex. Model: claude-sonnet-5-5 through the LiteLLM proxy (Anthropic format, ADK `AnthropicLlm` with a proxy-bound client).
- Tracing: OpenInference `GoogleADKInstrumentor`. It also disables ADK's native tracers, so each model call yields one LLM span.
- The Anthropic client is built with `open_telemetry=False`; anthropic 2.8 otherwise adds its own `anthropic.messages.create` span, which duplicates each GENERATION.
- `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT=4096` is set in `adk_common.py`. With the default of 128, long conversations drop `llm.model_name` and Langfuse cannot price the generation.
- A per-run LLM-call budget is enforced by an ADK plugin (`adk_common.CallBudgetPlugin`), which also covers AgentTool child runs.

## Tools

web_search, fetch_url

## Cost

Smoke run (batch smoke, --limit 3, 2026-11-12): 3/3 CHECK: PASS.
Mean 14.3 LLM generations per run.
Mean cost $0.1874 per run (Langfuse trace cost).
