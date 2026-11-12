# raw-reflexion-coder

Framework tag: `raw-anthropic`. Tracing: manual (Langfuse child observations for phases and tools; OpenInference anthropic instrumentor for LLM generations).

## Source

https://github.com/noahshinn/reflexion/blob/main/programming_runs/reflexion.py

Reflexion's programming loop: internal test generation, implementation, test execution, verbal self-reflection, re-implementation (prompts from generators/py_generate.py).

## What was adapted

- Local executor replaced by the tools service's sandboxed python_exec.
- Spec and tests come from one call (signature + asserts). All reflections so far are kept as memory for the next attempt. max_iters=3, pass@1.
- No hidden benchmark tests: the result reports the self-generated test outcome.
- A Sonnet `stop_reason == "refusal"` (safety classifier, e.g. on a destructive-code request) ends the run with a refusal message; the span is marked.

## Tools

python_exec

## Trace shape

agent > spec_and_internal_tests; attempt-N > [self_reflection (evaluator)] + implement + python_exec tool

## Smoke cost

Smoke batch `smoke`, first 3 tasks, 3/3 `CHECK: PASS` (2026-11-12).
Mean LLM calls per run: 1.3. Mean cost per run: USD 0.0041 (Langfuse calculated cost). The smoke set has a refusal (t002) and an edge task that stops after the spec call (t003). A full run that needs a reflection costs about USD 0.013-0.026 (2-4 Sonnet calls).
