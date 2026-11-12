# lg-supervisor-team

Supervisor routes between research, math and writer react sub-agents via handoff tools.

- Source: https://github.com/langchain-ai/langgraph-supervisor-py
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: medium.

## Derived from

Hierarchical supervisor (langchain-ai/langgraph-supervisor-py quickstart), extended to 3 workers.

## What was adapted

Kept from the source: `create_supervisor` over `create_react_agent` workers, the tool-based
handoff mechanism (transfer_to_<agent> / transfer_back_to_supervisor), the "research expert does
no math" split, and the "use one tool at a time" math prompt.
Changed: the mocked web_search is replaced by real keyless search/Wikipedia/arXiv, the math expert
gains sandboxed Python, and a writer worker (the README's 3-agent variant) formats the deliverable.

## Tools

research_expert: web_search, fetch_url, wikipedia_search, arxiv_search; math_expert: local add/multiply/divide/percent_change/compound_growth + python_exec; writer: local word_count

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [7, 5, 11] (mean 7.7). Cost per run from Langfuse: mean $0.0069 (total $0.0206 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.
