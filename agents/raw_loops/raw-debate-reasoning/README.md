# raw-debate-reasoning

Framework tag: `raw-litellm`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference litellm instrumentor for LLM generations).

## Source

https://github.com/composable-models/llm_multiagent_debate/blob/main/gsm/gen_gsm.py

Multi-agent debate (Du et al.): 3 agents, 2 rounds, other agents' answers fed back, \boxed{} answers.

## What was adapted

- Uses the LiteLLM Python SDK (`litellm.acompletion`, model openai/claude-haiku-5-5 via the proxy).
- Debaters in a round run concurrently. A majority_vote local tool runs after every round; a third round runs only without a majority.
- Answers generalised from numbers to short answers so non-math questions work.

## Tools

local: majority_vote

## Trace shape

agent > debate-round-N (chain) > debater-1..3 (agent) > litellm generation; majority_vote tool

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 6.0. Mean cost per run: USD 0.0013 (Langfuse calculated cost).
