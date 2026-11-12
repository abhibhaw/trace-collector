# ag-writer-critic

Writer/critic reflection loop on AutoGen AgentChat RoundRobinGroupChat.

- Framework: autogen
- Source: https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/tutorial/teams.ipynb
- Complexity: simple
- Model: claude-haiku-5-5
- Tracing: OpenInference AutoGen instrumentor (team, agent, tool spans) plus OpenInference Anthropic instrumentor (LLM generations)

## What was adapted

Derived from the AgentChat tutorial "Teams" (primary + critic agents, TextMentionTermination("APPROVE")).
Adapted: the writer can ground copy with web search, Wikipedia and the Lumen product KB; the loop is
bounded with MaxMessageTermination; model is Claude via the LiteLLM proxy.

## Tools

- `kb_search` (peira_harness.tools)
- `web_search` (peira_harness.tools)
- `wikipedia_search` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [7, 8, 10] (mean 8.3).
Cost per run from Langfuse: ['$0.0042', '$0.0028', '$0.0077'] (mean $0.0049).
Projected cost for 100 tasks: about $0.49.
