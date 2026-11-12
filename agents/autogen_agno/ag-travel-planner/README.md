# ag-travel-planner

Travel planning team on AutoGen AgentChat SelectorGroupChat.

- Framework: autogen
- Source: https://github.com/microsoft/autogen/blob/main/python/docs/src/user-guide/agentchat-user-guide/examples/travel-planning.ipynb
- Complexity: medium
- Model: claude-haiku-5-5
- Tracing: OpenInference AutoGen instrumentor (team, agent, tool spans) plus OpenInference Anthropic instrumentor (LLM generations)

## What was adapted

Derived from the AgentChat "Travel Planning" example (planner, local, language and summary agents).
Adapted: the round-robin order is replaced by SelectorGroupChat with the selector prompt and a
candidate_func from the "Selector Group Chat" guide; a travel agent with flight and hotel search tools
and a company travel-policy lookup was added; local_agent grounds suggestions with Wikipedia and web search.

## Tools

- `hr_policy_lookup` (peira_harness.tools)
- `travel_search_flights` (peira_harness.tools)
- `travel_search_hotels` (peira_harness.tools)
- `web_search` (peira_harness.tools)
- `wikipedia_search` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [22, 25, 26] (mean 24.3).
Cost per run from Langfuse: ['$0.0117', '$0.0171', '$0.0125'] (mean $0.0138).
Projected cost for 100 tasks: about $1.38.
