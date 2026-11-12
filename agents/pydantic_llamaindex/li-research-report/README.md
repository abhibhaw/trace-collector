# li-research-report

Framework: LlamaIndex.
Source: https://github.com/run-llama/llama_index/blob/main/docs/examples/agent/agent_workflow_multi.ipynb

## Adaptation

- Kept: `AgentWorkflow` with ResearchAgent, WriteAgent, and ReviewAgent `FunctionAgent`s, their prompts and `can_handoff_to` graph, and the `record_notes` / `write_report` / `review_report` tools that edit `Context` state.
- Changed: Tavily is replaced by keyless research tools. Runs on `claude-sonnet-5-5` (auto tool choice only). `max_iterations=18`.

## Tools

web_search, wikipedia_search, wikipedia_page, arxiv_search, fetch_url

## Tracing

OpenInference `LlamaIndexInstrumentor`. The LLM is `pl_common.ProxyAnthropic`: it removes the dispatcher spans from LLM helper methods that make no API call (`_prepare_chat_with_tools`, `apredict`, `acomplete`, ...), so Langfuse shows exactly one GENERATION per API call and cost is not counted twice.
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [7, 9, 9], mean 8.3.
Cost per run from Langfuse (USD): [0.092948, 0.17966, 0.212654], mean 0.161754.
Estimated full batch (100 tasks): about USD 16.18.
