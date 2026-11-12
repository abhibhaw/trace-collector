# crew-game-builder

Source: https://github.com/crewAIInc/crewAI-examples/tree/main/crews/game-builder-crew

You are helping create a game using python, these are the instructions:\n\n

## What was adapted

Game builder crew, adapted from crewAI-examples/crews/game-builder-crew.

Shape kept from the source: senior engineer writes the game, QA engineer reviews it, chief QA engineer
(allow_delegation=True) evaluates it and may delegate fixes back to coworkers. Sequential process.
Adapted: every engineer can run code in the sandboxed python_exec tool. The sandbox has no stdin, so each
game must include a scripted, non-interactive demo that exercises the game logic.

## Facts

- Framework: CrewAI 1.15.40, native Anthropic provider through the LiteLLM proxy.
- Tracing: OpenInference CrewAI instrumentor in event-listener mode (crew, agent, task, tool and LLM spans; one LLM span per call, with token usage).
- Model: claude-haiku-5-5.
- Complexity: medium.
- Shared tools: python_exec.
- Local tools: none.

## Smoke run cost

Smoke batch `smoke`, 3 tasks, 3/3 `CHECK: PASS` in `scripts/inspect_trace.py`.
Mean LLM calls (GENERATION spans) per run: 6.3 (per task: 6, 6, 7).
Mean cost per run: $0.0075 (Langfuse trace cost).
