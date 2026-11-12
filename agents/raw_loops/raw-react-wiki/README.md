# raw-react-wiki

Framework tag: `raw-litellm`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference litellm instrumentor for LLM generations).

## Source

https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb

The original ReAct `webthink` loop with the 6-shot webthink_simple6 prompt (copied to webthink_simple6.txt, MIT licence) and wikienv's Search/Lookup/Finish actions.

## What was adapted

- Text-parsed actions with stop sequences, no native tool calling. Uses the LiteLLM Python SDK (`litellm.completion`) through the proxy.
- Completion API replaced by chat: a system message tells the model to continue the trajectory.
- WikiEnv ported onto wikipedia_page / wikipedia_search; Lookup runs over the cached page. Max 7 steps; raises if no Finish.

## Tools

wikipedia_page, wikipedia_search (wrapped as the search / lookup env actions)

## Trace shape

agent > step-N (chain) > litellm generation(s) + search/lookup tool span

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 3.7. Mean cost per run: USD 0.0014 (Langfuse calculated cost).
