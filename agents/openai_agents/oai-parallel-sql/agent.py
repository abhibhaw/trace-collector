"""Parallel SQL analysts + picker: the same analyst agent runs three times in parallel on a
data question over the sample SQLite databases, then a picker agent chooses the best answer.

Adapted from openai-agents-python examples/agent_patterns/parallelization.py. The source runs a
Spanish translator three times with asyncio.gather and a translation_picker chooses the best.
Here the parallel workers are SQL analysts with schema-inspection and query tools, which makes
the candidates genuinely differ (different queries, joins, and interpretations).
"""

import asyncio

from agents import Agent, ItemHelpers, Runner, trace

from _oai_common import HAIKU, model, refusal_as_answer, settings, setup_family, shared_tools
from peira_harness import AgentMeta

AGENT = AgentMeta(
    id="oai-parallel-sql",
    framework="openai-agents",
    source_url="https://github.com/openai/openai-agents-python/blob/main/examples/agent_patterns/parallelization.py",
    description="Three parallel SQL analyst agents answer a data question over Chinook/Northwind, then a picker agent selects and returns the best answer.",
    complexity="medium",
    model=HAIKU,
    tracing="openinference",
)

N_PARALLEL = 3
SQL_TOOLS: tuple[str, ...] = ("sql_schema", "sql_query")

_a: dict = {}


def setup() -> None:
    setup_family()
    _a["analyst"] = Agent(
        name="sql_analyst",
        instructions=(
            "You answer data questions with read-only SQL over the sample SQLite databases "
            "(chinook: a digital music store; northwind: a food trading company). Inspect the schema "
            "before writing queries. Make at most 4 tool calls. Report the final SQL you used, the "
            "result, and a one-paragraph answer. If the question is ambiguous or cannot be answered "
            "from the data, say exactly why. Never run statements that modify data."
        ),
        model=model(HAIKU),
        model_settings=settings(),
        tools=shared_tools(*SQL_TOOLS),
    )
    _a["picker"] = Agent(
        name="answer_picker",
        instructions=(
            "You pick the best answer to a data question from the given candidate answers. Prefer "
            "the candidate whose SQL is correct and whose number is supported by its query result. "
            "If candidates disagree, say which you trust and why. Return the chosen answer, cleaned up, "
            "including its SQL."
        ),
        model=model(HAIKU),
        model_settings=settings(),
    )


@refusal_as_answer
async def arun(task: dict) -> str:
    msg = task["input"]
    with trace("Parallel SQL answers"):
        results = await asyncio.gather(
            *(Runner.run(_a["analyst"], msg, max_turns=6) for _ in range(N_PARALLEL)),
            return_exceptions=True,
        )
        ok = [r for r in results if not isinstance(r, BaseException)]
        if not ok:
            raise results[0]
        outputs = [ItemHelpers.text_message_outputs(r.new_items) or str(r.final_output) for r in ok]
        candidates = "\n\n".join(f"Candidate {i + 1}:\n{o}" for i, o in enumerate(outputs))
        best = await Runner.run(_a["picker"], f"Question: {msg}\n\nCandidates:\n{candidates}", max_turns=2)
    return str(best.final_output)
