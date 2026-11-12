# lg-plan-execute

Planner -> ReAct step executor -> replanner loop with structured Plan/Act outputs.

- Source: https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: medium.

## Derived from

Plan-and-execute (LangGraph tutorial), adapted to the peira proxy and tools.

## What was adapted

Kept from the source: PlanExecute state (input / plan / past_steps with operator.add / response),
the planner and replanner prompts verbatim, the Plan / Response / Act structured outputs, a
prebuilt ReAct executor that runs step 1 of the current plan, and the planner -> agent -> replan
loop with `should_end`.
Changed: Tavily is replaced by keyless web search, Wikipedia and sandboxed Python. A step cap
bounds cost: after MAX_EXECUTED_STEPS executed steps, the replanner must respond.

## Tools

executor: web_search, fetch_url, wikipedia_search, python_exec

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [3, 12, 3] (mean 6.0). Cost per run from Langfuse: mean $0.0020 (total $0.0061 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.
