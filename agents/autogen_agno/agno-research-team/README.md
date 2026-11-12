# agno-research-team

Research team on Agno Team in coordinate mode.

- Framework: agno
- Source: https://github.com/agno-agi/agno/blob/main/cookbook/03_teams/02_modes/coordinate/02_with_tools.py
- Complexity: medium
- Model: claude-haiku-5-5
- Tracing: OpenInference Agno instrumentor (agent, team, workflow, tool and LLM spans)

## What was adapted

Derived from the Agno cookbook "Coordinate Mode with Tools" (team leader delegates to member agents with
specialized tools and synthesizes their findings). Adapted: HackerNews/DuckDuckGo members are replaced by
a Web Searcher (keyless web search + URL fetch), a Wikipedia Researcher and an arXiv Researcher; the
leader and members use Claude through the LiteLLM proxy; tool calls per member are capped.

## Tools

- `arxiv_search` (peira_harness.tools)
- `fetch_url` (peira_harness.tools)
- `web_search` (peira_harness.tools)
- `wikipedia_page` (peira_harness.tools)
- `wikipedia_search` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [4, 4, 10] (mean 6.0).
Cost per run from Langfuse: ['$0.0011', '$0.0036', '$0.0038'] (mean $0.0028).
Projected cost for 100 tasks: about $0.28.
