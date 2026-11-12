# dspy-multihop-rag

Multi-hop Wikipedia RAG: per hop a CoT writes a query, retrieval runs, and a CoT appends notes; a final CoT answers.

- Framework: dspy
- Source: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/multihop_search/index.ipynb
- Model: claude-haiku-5-5
- Complexity: medium
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from the DSPy "Tutorial: Multi-Hop Retrieval" (docs/docs/tutorials/multihop_search): the `Hop`
module with two ChainOfThought sub-modules, `generate_query` ('claim, notes -> query') and
`append_notes` ('claim, notes, context -> new_notes: list[str], titles: list[str]'), run for num_hops.
Changed: the local BM25 index over 2017 wiki abstracts is replaced by the tools-service Wikipedia search;
num_hops is 3 (source: 4) to bound cost; a final ChainOfThought turns the notes into an answer;
no optimization is run.

## Tools

wikipedia_search (tools service) as the retriever

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 7.0 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0017 (Langfuse per-trace cost); smoke total $0.0052.
