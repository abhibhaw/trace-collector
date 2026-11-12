# crew-book-flow

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/flows/write_a_book_with_flows

Write a well-structured chapter based on the chapter title, goal, and outline description. Write it 

## What was adapted

Write-a-book flow, adapted from crewAI-examples/flows/write_a_book_with_flows.

Shape kept from the source: a Flow with BookState; @start runs the OutlineCrew (researcher -> outliner, pydantic
BookOutline); write_chapters runs one WriteBookChapterCrew per chapter concurrently (researcher -> writer,
pydantic Chapter); join_and_save_chapter assembles the markdown book.
Adapted: SerperDevTool is replaced with web search and Wikipedia; topic and goal come from the request; to bound
cost the outline is capped at 3 chapters of about 400-600 words; a @router after the outline skips chapter
writing when the outliner returns no usable chapters (refused or unclear topic); the book is returned, not saved.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: complex.
- Shared tools: web_search, wikipedia_page, wikipedia_search.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 11.7 (per task: 2, 17, 16).
Mean cost per run: $0.0166 (Langfuse trace cost).
Task t003 is adversarial and routes to the no-outline branch (2 calls).
