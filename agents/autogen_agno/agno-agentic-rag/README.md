# agno-agentic-rag

Agentic RAG over the Lumen product knowledge base on Agno.

- Framework: agno
- Source: https://github.com/agno-agi/agno/blob/main/cookbook/07_knowledge/04_advanced/01_custom_retriever.py
- Complexity: simple
- Model: claude-haiku-5-5
- Tracing: OpenInference Agno instrumentor (agent, team, workflow, tool and LLM spans)

## What was adapted

Derived from the Agno cookbook "Custom Retriever" (Agent with knowledge_retriever instead of a vector-DB
Knowledge object; Agno exposes it to the model as the search_knowledge_base tool) and "Agentic RAG"
(the agent decides when and how often to search). Adapted: the retriever calls the tools-service KB
search (no OpenAI embeddings, no vector DB), and the agent can open a full article by id.

## Tools

- `kb_get_article` (peira_harness.tools)
- `kb_search` (peira_harness.tools)

## Smoke run cost

Smoke batch `smoke`, tasks t001-t003 (normal, edge, adversarial): 3/3 `inspect_trace.py` PASS.
LLM calls per run: [2, 2, 3] (mean 2.3).
Cost per run from Langfuse: ['$0.0005', '$0.0006', '$0.0007'] (mean $0.0006).
Projected cost for 100 tasks: about $0.06.
