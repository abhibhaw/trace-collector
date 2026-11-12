# pyd-sql-gen

Framework: PydanticAI.
Source: https://github.com/pydantic/pydantic-ai/blob/main/examples/pydantic_ai_examples/sql_gen.py

## Adaptation

- Kept: schema and few-shot examples in dynamic instructions, the `Success | InvalidRequest` union output, and the `@output_validator` that rejects non-SELECT SQL and raises `ModelRetry` with the database error.
- Changed: Postgres `EXPLAIN` became a 1-row test run through the tools service. The schemas of Chinook and Northwind are loaded in `setup()`.
- Added: after validation the query runs and an `answer_agent` writes the answer, sharing the run's usage and `UsageLimits`.

## Tools

sql_schema (setup), sql_query

## Tracing

PydanticAI built-in OpenTelemetry instrumentation (`Agent.instrument_all`, global tracer provider).
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [1, 2, 2], mean 1.7.
Cost per run from Langfuse (USD): [0.000605, 0.000724, 0.001115], mean 0.000815.
Estimated full batch (100 tasks): about USD 0.08.
