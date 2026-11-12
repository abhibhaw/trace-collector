"""Multi-agent debate (Du et al., 2023) over the LiteLLM Python SDK.

Adapted from composable-models/llm_multiagent_debate (gsm/gen_gsm.py): N agents answer independently,
then in each round every agent reads the other agents' latest answers and revises its own.
Answers are \\boxed{...}; a majority vote (local tool) picks the result. If round 2 ends without a
majority, one extra round runs. Rounds are chain spans, each debater turn is an agent span.
"""
from __future__ import annotations

import asyncio
import re
from collections import Counter

import litellm

from peira_harness import AgentMeta, llm
from raw_common import begin_task, Toolbox, instrument_litellm, lit_text, phase

AGENT = AgentMeta(
    id="raw-debate-reasoning",
    framework="raw-litellm",
    source_url="https://github.com/composable-models/llm_multiagent_debate/blob/main/gsm/gen_gsm.py",
    description="Three LLM agents debate a reasoning question over two or three rounds, then a majority vote picks the answer.",
    complexity="medium",
    model=llm.HAIKU,
    tracing="manual",
)

N_AGENTS = 3
ROUNDS = 2
MAX_ROUNDS = 3


def setup() -> None:
    litellm.drop_params = True
    instrument_litellm()


INITIAL = (
    "Can you answer the following question? {question} Explain your reasoning. "
    "Your final answer should be a short answer (a single number, word, or phrase), "
    "in the form \\boxed{{answer}}, at the end of your response. If the question cannot be answered as asked, "
    "explain why and box the word unanswerable."
)

DEBATE = (
    "These are the solutions to the problem from other agents: {others}\n\n"
    "Using the reasoning from other agents as additional information, can you provide your answer to the problem? "
    "Check their work critically; do not simply follow the majority. "
    "The original problem is: {question} Your final answer should be a short answer in the form \\boxed{{answer}}, "
    "at the end of your response."
)


def boxed(text: str) -> str:
    hits = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", text or "")
    return hits[-1].strip() if hits else ""


def normalize(ans: str) -> str:
    a = ans.strip().lower()
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace("$", "").replace(",", "").replace("\\%", "%").strip(" .")
    if re.fullmatch(r"-?\d+\.0+", a):
        a = a.split(".")[0]
    return a


def majority_vote(answers: list[str]) -> dict:
    """Majority vote over the agents' boxed answers after light normalisation.

    Args:
        answers: One final boxed answer per agent (empty string if the agent gave none).
    """
    norm = [normalize(a) for a in answers if a.strip()]
    if not norm:
        return {"winner": None, "counts": {}, "consensus": False}
    counts = Counter(norm)
    winner, n = counts.most_common(1)[0]
    return {"winner": winner, "counts": dict(counts), "consensus": n > len(answers) / 2, "unanimous": n == len(answers)}


TOOLS = Toolbox([majority_vote])


async def ask(context: list[dict], name: str, round_no: int) -> str:
    with phase(name, "agent", input=context[-1]["content"], metadata={"round": round_no}) as a:
        resp = await litellm.acompletion(
            model=llm.litellm_model_name(AGENT.model),
            api_base=llm.openai_base_url(),
            api_key=llm.proxy_key(),
            max_tokens=llm.MAX_TOKENS,
            messages=context,
            num_retries=3,
        )
        text = lit_text(resp)
        a.update(output={"boxed": boxed(text)})
    context.append({"role": "assistant", "content": text})
    return text


async def arun(task: dict) -> str:
    begin_task(AGENT.id, task)
    question = task["input"]
    contexts = [[{"role": "user", "content": INITIAL.format(question=question)}] for _ in range(N_AGENTS)]
    vote: dict = {}
    for rnd in range(MAX_ROUNDS):
        if rnd >= ROUNDS and vote.get("consensus"):
            break
        with phase(f"debate-round-{rnd + 1}", "chain") as r:
            if rnd > 0:
                latest = [ctx[-1]["content"] for ctx in contexts]
                for i, ctx in enumerate(contexts):
                    others = "".join(f"\n\n One agent solution: ```{t}```" for j, t in enumerate(latest) if j != i)
                    ctx.append({"role": "user", "content": DEBATE.format(others=others, question=question)})
            replies = await asyncio.gather(*(ask(ctx, f"debater-{i + 1}", rnd + 1) for i, ctx in enumerate(contexts)))
            answers = [boxed(t) for t in replies]
            text, _ = TOOLS.call("majority_vote", {"answers": answers})
            vote = majority_vote(answers)
            r.update(output={"answers": answers, "vote": text})

    winner = vote.get("winner")
    if not winner:
        raise RuntimeError("no debater produced a boxed answer")
    best = next(ctx[-1]["content"] for ctx in contexts if normalize(boxed(ctx[-1]["content"])) == winner)
    status = "unanimous" if vote.get("unanimous") else ("majority" if vote.get("consensus") else "no majority; plurality")
    return f"Answer: {boxed(best)} ({status} of {N_AGENTS} debaters)\n\n{best}"
