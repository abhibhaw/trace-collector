# dspy-math-rlm

dspy.RLM math solver that reasons by writing and running Python in a sandboxed REPL, with an SQL tool for data questions.

- Framework: dspy
- Source: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/math/index.ipynb
- Model: claude-haiku-5-5
- Complexity: simple
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from the DSPy "Tutorial: Math Reasoning" (docs/docs/tutorials/math), which builds
`dspy.ChainOfThought("question -> answer")` and recommends a code sandbox for harder reasoning, and from the
RLM migration guide (docs/docs/api/modules/RLM.md): `dspy.ProgramOfThought` and `dspy.CodeAct` were removed in
DSPy 3.5 and `dspy.RLM("question -> answer", tools=[...])` is the documented replacement.
The RLM writes Python in a Deno/Pyodide sandbox (no host env or files), can call `llm_query`, and SUBMITs.
A host-side `sql_query` tool (tools service, Chinook/Northwind) is exposed for data-backed word problems.
Bounded: max_iters=6, max_llm_calls=3. No optimization (MIPROv2) is run.

## Tools

Deno/Pyodide Python sandbox (dspy.PythonInterpreter, local to the pod); sql_query, sql_schema (tools service, host-side RLM tools)

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 2.0 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0007 (Langfuse per-trace cost); smoke total $0.0022.
