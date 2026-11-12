# li-router-qe

Framework: LlamaIndex.
Source: https://github.com/run-llama/llama_index/blob/main/docs/examples/workflow/router_query_engine.ipynb

## Adaptation

- Kept: `RouterQueryEngineWorkflow` with `selector` (LLM multi-selector over `QueryEngineTool` metadata), `generate_responses` (parallel `aquery`), and `synthesize_responses` (`TreeSummarize` when more than one engine answered).
- Changed: the notebook's vector and summary indexes (OpenAI embeddings) are replaced by four `RetrieverQueryEngine`s over custom tool-backed retrievers: product KB, HR policies, market data, Wikipedia. No embeddings.

## Tools

kb_search, kb_get_article, hr_policy_lookup, finance_quote, wikipedia_search, wikipedia_page

## Tracing

OpenInference `LlamaIndexInstrumentor`. The LLM is `pl_common.ProxyAnthropic`: it removes the dispatcher spans from LLM helper methods that make no API call (`_prepare_chat_with_tools`, `apredict`, `acomplete`, ...), so Langfuse shows exactly one GENERATION per API call and cost is not counted twice.
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [2, 2, 2], mean 2.0.
Cost per run from Langfuse (USD): [0.000294, 0.00051, 0.000422], mean 0.000409.
Estimated full batch (100 tasks): about USD 0.04.
