"""LLM-as-a-judge loop: a copywriter agent drafts product copy, an evaluator agent scores it
with structured feedback, and the loop repeats with the feedback until it passes.

Adapted from openai-agents-python examples/agent_patterns/llm_as_a_judge.py. The source loops a
story-outline generator against an evaluator. Here the generator writes Lumen & Co product copy
grounded in the product catalog tool, and the evaluator checks it against the same catalog.
The loop is capped at 3 rounds (the source caps auto mode at 3).
"""
from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, Runner, TResponseInputItem, trace

from _oai_common import HAIKU, model, refusal_as_answer, settings, setup_family, shared_tools
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="oai-judge-loop",
    framework="openai-agents",
    source_url="https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/llm_as_a_judge.py",
    description="Copywriter agent and evaluator agent loop with structured pass/needs_improvement/fail feedback until the copy passes (max 3 rounds).",
    complexity="medium",
    model=HAIKU,
    tracing="openinference",
)

MAX_ROUNDS = 3


@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]


_a: dict = {}


def setup() -> None:
    setup_family()
    catalog = shared_tools(*PRODUCT_TOOLS)
    _a["generator"] = Agent(
        name="copy_generator",
        instructions=(
            "You write short product or marketing copy for Lumen & Co, a smart-home device maker, "
            "based on the user's input. Look up the real product facts (price, specs, warranty, stock) "
            "with the catalog tools first and never invent specs. "
            "If there is any feedback provided, use it to improve the copy."
        ),
        model=model(HAIKU),
        model_settings=settings(),
        tools=catalog,
    )
    _a["evaluator"] = Agent[None](
        name="evaluator",
        instructions=(
            "You evaluate a piece of product copy and decide if it's good enough. "
            "Check the facts against the catalog tools, the requested tone and length, and any brand "
            "constraints in the request. If it's not good enough, you provide feedback on what needs "
            "to be improved. Do not give it a pass on the first try unless it is clearly excellent. "
            "After 2 attempts, you can give it a pass if the copy is good enough - do not go for perfection. "
            "Use 'fail' only if the request itself cannot be satisfied truthfully."
        ),
        model=model(HAIKU),
        model_settings=settings(),
        tools=catalog,
        output_type=EvaluationFeedback,
    )


PRODUCT_TOOLS: tuple[str, ...] = ("products_search", "inventory_check")


@refusal_as_answer
async def arun(task: dict) -> str:
    items: list[TResponseInputItem] = [{"content": task["input"], "role": "user"}]
    latest = ""
    result: EvaluationFeedback | None = None
    with trace("LLM as a judge"):
        for rnd in range(1, MAX_ROUNDS + 1):
            gen = await Runner.run(_a["generator"], items, max_turns=4)
            items = gen.to_input_list()
            latest = ItemHelpers.text_message_outputs(gen.new_items) or str(gen.final_output)
            # Claude rejects a conversation ending with an assistant turn (no prefill): ask explicitly.
            ev_input = items + [{"role": "user", "content": "Evaluate the latest copy above against the request."}]
            ev = await Runner.run(_a["evaluator"], ev_input, max_turns=3)
            result = ev.final_output
            if result.score in ("pass", "fail") or rnd == MAX_ROUNDS:
                break
            items.append({"content": f"Feedback: {result.feedback}", "role": "user"})
    return f"{latest}\n\n[judge: {result.score} after {rnd} round(s)] {result.feedback}"
