# li-crag-workflow

Framework: LlamaIndex.
Source: https://github.com/run-llama/llama_index/blob/main/docs/examples/workflow/corrective_rag_pack.ipynb

## Adaptation

- Kept: the event-driven `Workflow` (`PrepEvent`, `RetrieveEvent`, `RelevanceEvalEvent`, `TextExtractEvent`, `QueryEvent`), the per-document yes/no relevance grader and query-rewrite prompts verbatim, and the final `SummaryIndex` query engine.
- Changed: the vector index is replaced by `kb_search` + `kb_get_article` (at most 4 graded docs). Tavily is replaced by `web_search`.
- Added: a corrective loop (`RetryRetrieveEvent`) that retries the KB once with the rewritten query when no document was relevant.

## Tools

kb_search, kb_get_article, web_search

## Tracing

OpenInference `LlamaIndexInstrumentor`. The LLM is `pl_common.ProxyAnthropic`: it removes the dispatcher spans from LLM helper methods that make no API call (`_prepare_chat_with_tools`, `apredict`, `acomplete`, ...), so Langfuse shows exactly one GENERATION per API call and cost is not counted twice.
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [6, 6, 11], mean 7.7.
Cost per run from Langfuse (USD): [0.000957, 0.000929, 0.002058], mean 0.001315.
Estimated full batch (100 tasks): about USD 0.13.
