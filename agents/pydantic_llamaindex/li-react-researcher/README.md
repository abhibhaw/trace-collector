# li-react-researcher

Framework: LlamaIndex.
Source: https://github.com/run-llama/llama_index/blob/main/docs/examples/agent/react_agent.ipynb

## Adaptation

- Kept: `ReActAgent` with the notebook's local calculator tools (`multiply`, `add`, plus `subtract`, `divide`) and a custom system prompt.
- Added: research tools so questions combine lookups with arithmetic. `max_iterations=10`.

## Tools

wikipedia_search, wikipedia_page, arxiv_search, web_search, python_exec, local calculators

## Tracing

OpenInference `LlamaIndexInstrumentor`. The LLM is `pl_common.ProxyAnthropic`: it removes the dispatcher spans from LLM helper methods that make no API call (`_prepare_chat_with_tools`, `apredict`, `acomplete`, ...), so Langfuse shows exactly one GENERATION per API call and cost is not counted twice.
The Anthropic SDK's own OTel spans are off (`ANTHROPIC_OPEN_TELEMETRY=false`), so the framework is the only LLM span source.
Shared helpers are in `../pl_common.py`.

## Smoke cost

Smoke run on 2026-11-12, batch `smoke`, first 3 tasks, 3/3 `inspect_trace.py` PASS.
LLM calls (GENERATIONs) per run: [1, 2, 3], mean 2.0.
Cost per run from Langfuse (USD): [0.00045, 0.001112, 0.001754], mean 0.001105.
Estimated full batch (100 tasks): about USD 0.11.
