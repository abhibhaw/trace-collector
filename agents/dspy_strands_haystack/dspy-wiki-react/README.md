# dspy-wiki-react

dspy.ReAct agent that browses Wikipedia to find the pages needed to verify or refute a claim.

- Framework: dspy
- Source: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/agents/index.ipynb
- Model: claude-haiku-5-5
- Complexity: simple
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from the DSPy "Tutorial: Agents" (docs/docs/tutorials/agents): a dspy.ReAct with the two tools
`search_wikipedia` and `lookup_wikipedia`, instructed to find all Wikipedia titles needed to verify a claim.
Changed: the ColBERTv2 wiki17-abstracts server is replaced by the tools-service Wikipedia search/page tools,
the signature also asks for a short verdict, max_iters is 8 (source: 20), and no optimization is run.

## Tools

wikipedia_search, wikipedia_page (tools service)

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 2.3 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0010 (Langfuse per-trace cost); smoke total $0.0030.
