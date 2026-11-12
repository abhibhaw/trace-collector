# lg-reflexion-writer

Reflexion actor: draft answer with self-critique, run search queries, revise with citations.

- Source: https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/reflexion/reflexion.ipynb
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: simple.

## Derived from

Reflexion (LangGraph tutorial, Shinn et al.), adapted to the peira proxy and tools.

## What was adapted

Kept from the source: the actor prompt, AnswerQuestion / Reflection / ReviseAnswer schemas,
ResponderWithRetries (schema validation with a corrective ToolMessage), the run_queries ToolNode
registered under both function names, and the draft -> execute_tools -> revise loop that stops
after MAX_ITERATIONS trailing ai/tool messages.
Changed: Tavily is replaced by keyless web search (plus Wikipedia), the actor uses forced tool
choice for the answer schema, and MAX_ITERATIONS is lowered to bound cost (1 draft + 2 revisions).

## Tools

web_search (run_queries, registered as AnswerQuestion and ReviseAnswer like the source)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [5, 3, 5] (mean 4.3). Cost per run from Langfuse: mean $0.0038 (total $0.0114 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.
