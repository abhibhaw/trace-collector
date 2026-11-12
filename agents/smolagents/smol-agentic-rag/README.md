# smol-agentic-rag

Source: https://github.com/huggingface/smolagents/blob/main/examples/rag.py (and docs/source/en/examples/rag.md)

## Shape

One `CodeAgent` with a `RetrieverTool(Tool)` subclass (lexical KB search), a full-article reader and a Wikipedia retriever. A `final_answer_checks` validator forces KB citations.

## What was adapted

- The BM25 index over Hugging Face docs becomes the tools-service product knowledge base (`kb_search`, `kb_get_article`).
- A Wikipedia retriever covers general concepts.
- New: the `cites_kb_articles` final-answer check. It rejects answers that do not cite a retrieved `KB-xxxx` id, so the agent takes an extra step.

## Tools

`kb_search`, `kb_get_article`, `wikipedia_search`.

## Cost bound

`max_steps=5` (source: 4). Worst case: 6 LLM calls.

## Smoke run

- Smoke result: 3/3 PASS
- Mean LLM calls per run: 3.0
- Mean cost per run: $0.0018 (Langfuse trace cost on the 3 smoke tasks)

## Family notes (smolagents)

- Shared code is in `agents/smolagents/smol_common.py`.
- Model: `ProxyModel`, a `LiteLLMModel` subclass, calls `anthropic/<model>` at the proxy's Anthropic endpoint with `max_tokens=4096`.
  It changes smolagents' default `tool_choice="required"` to `"auto"`, because `claude-sonnet-5-5` rejects forced tool choice.
  It also removes `temperature`, `top_p` and `top_k`.
- Tracing: `SmolagentsInstrumentor` (OpenInference) is the only source of LLM spans.
  It creates agent, step, tool and LLM spans with token counts.
- A fresh agent is built for each task, because agents keep memory.
  Each task calls `tools.set_run_id(...)`, so that writes stay scoped to the run.
