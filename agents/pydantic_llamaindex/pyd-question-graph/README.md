# pyd-question-graph

Framework: PydanticAI + pydantic-graph.
Source: https://github.com/pydantic/pydantic-ai/blob/v1.0.0/examples/pydantic_ai_examples/question_graph.py

## Adaptation

- Kept: the `Ask -> Answer -> Evaluate -> Reprimand -> Ask` BaseNode state machine, per-agent message history in graph state, and `EvaluationOutput`.
- Changed: the human `input()` is a student agent with web tools. Ask and Evaluate verify facts in Wikipedia. The loop stops after 3 rounds.
- Changed: built with the pydantic-graph 2.x `GraphBuilder` (`g.node(...)`), because v2 removed `Graph(nodes=...)` and file persistence. The example was removed from main after v1, so the source URL points at v1.0.0.
- pydantic-graph only emits node spans through logfire, so each node opens a plain OTel span (`node Ask`, ...).

## Tools

wikipedia_search, wikipedia_page, web_search

## Tracing

PydanticAI built-in OpenTelemetry instrumentation (`Agent.instrument_all`, global tracer provider).
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [4, 4, 4], mean 4.0.
Cost per run from Langfuse (USD): [0.001125, 0.001307, 0.001137], mean 0.001190.
Estimated full batch (100 tasks): about USD 0.12.
