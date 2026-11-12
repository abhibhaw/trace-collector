# hay-fallback-rag

Haystack pipeline: BM25 RAG over a Seven Wonders KB; a ConditionalRouter falls back to web search when the LLM says no_answer.

- Framework: haystack
- Source: https://github.com/deepset-ai/haystack-cookbook/blob/main/notebooks/llama32_agentic_rag.ipynb
- Model: claude-haiku-5-5
- Complexity: simple
- Tracing: openinference (see `../dsh_common.py`)

## What was adapted

Derived from deepset-ai/haystack-cookbook notebooks/llama32_agentic_rag.ipynb (and the tutorial it cites,
"Building Fallbacks to Websearch with Conditional Routing").
Kept: the Seven Wonders knowledge base (bilgeyucel/seven-wonders, 151 docs, bundled as seven_wonders.json),
the two prompt templates ('no_answer' / "FROM THE KNOWLEDGE BASE:" / "FROM THE WEB:"), the ConditionalRouter
routes, the BranchJoiner loop that sends the web-search prompt back into the same LLM, and the exact
component graph and connections.
Changed: Llama 3.2 (transformers) -> Claude via the proxy; sentence-transformers embedding retrieval ->
InMemoryBM25Retriever (no torch in the pod); DuckduckgoApiWebSearch -> a small component over the
tools-service web_search.

## Tools

InMemoryBM25Retriever over bundled seven_wonders.json (151 docs); web_search (tools service) in a custom WebSearch component

## Smoke run

Batch `smoke`, `--limit 3 --concurrency 3`, real models through the proxy (2026-11-12).

- inspect_trace: 3/3 PASS (nested spans, one GENERATION per LLM call, input and output usage on every generation).
- Mean LLM calls per run: 2.0 (models seen: claude-haiku-5-5).
- Mean cost per run: $0.0008 (Langfuse per-trace cost); smoke total $0.0024.
