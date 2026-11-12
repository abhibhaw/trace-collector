# raw-tot-number-puzzle

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/methods/bfs.py

Tree-of-Thoughts BFS for Game of 24 (propose prompt, value prompt with sure/likely/impossible, greedy selection).

## What was adapted

- Generalised to 2-6 numbers, any target, and banned operators.
- Breadth 3 instead of 5; all candidate states of one level are valued in a single batched call (the paper samples 3 value calls per state).
- Proposals are checked locally with exact fractions (guardrail span); the final expression is verified by a local check_expression tool; when the search fails, a local exhaustive_search tool says only whether a solution exists.
- States with 2 numbers left are valued exactly in code (one step from the target), and the batched value call gets max_tokens 12000, because Haiku's adaptive thinking can use all of 4096 on it.

## Tools

local: check_expression, exhaustive_search

## Trace shape

agent > parse_puzzle; puzzle_guardrail; bfs-level-N > generate (parallel proposals) + validate_proposals (guardrail) + evaluate (evaluator); final_answer > tool

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 9.3. Mean cost per run: USD 0.0009 (Langfuse calculated cost).
