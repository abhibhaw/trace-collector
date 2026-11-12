# lg-adaptive-rag

Routed RAG over the company KB or web, with document, hallucination and answer graders and a rewrite loop.

- Source: https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_adaptive_rag.ipynb
- Framework: langgraph. Tracing: langchain callbacks (Langfuse CallbackHandler).
- Model: claude-haiku-5-5. Complexity: medium.

## Derived from

Adaptive RAG with a corrective fallback (LangGraph adaptive-RAG + CRAG notebooks).

## What was adapted

Kept from the source: GraphState (question / generation / documents), the question router
(RouteQuery: vectorstore vs web_search), per-document relevance grading (GradeDocuments),
the rlm/rag-prompt generator, the hallucination grader and answer grader, the question
re-writer, and the edges START -> route, retrieve -> grade_documents -> (transform_query |
generate), generate -> (not supported -> generate | not useful -> transform_query | useful -> END).
Changed: the Chroma vectorstore over blog posts is replaced by the tools service knowledge base
(kb_search), Tavily by keyless web search + Wikipedia. Loops are capped with counters, and when
query rewriting is exhausted the graph takes the CRAG corrective branch to web search.

## Tools

kb_search + kb_get_article + hr_policy_lookup (vectorstore stand-in), web_search + wikipedia_search (web branch)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (first 3 tasks, which include edge and adversarial cases), 2026-11-12: 3/3 traces PASS `scripts/inspect_trace.py`. LLM calls per run: [24, 8, 8] (mean 13.3). Cost per run from Langfuse: mean $0.0017 (total $0.0052 for 3 runs). Normal tasks run longer than these edge cases, so expect higher means in the full batch.
