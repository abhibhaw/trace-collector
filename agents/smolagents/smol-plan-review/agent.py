"""Plan customization with a step callback, adapted from smolagents examples/plan_customization.

Source shape (examples/plan_customization/plan_customization.py):
    CodeAgent(tools=[search], planning_interval=..., step_callbacks={PlanningStep: interrupt_after_plan})
    The callback shows the plan to a human who can (1) approve, (2) modify `memory_step.plan`,
    or (3) cancel with `agent.interrupt()`. After a cancel, the run is resumed with `reset=False`
    so the agent keeps its memory.
Adaptation: the `input()` prompts are replaced by a deterministic auto-reviewer that simulates the
human. It approves short plans, edits long or unsourced plans (trims steps and adds review constraints),
and cancels plans that touch unsafe topics. On a cancel, the run resumes with reset=False and the
reviewer's feedback, as in the source's resume demonstration. DuckDuckGoSearchTool is replaced by
the tools-service web search, fetch, and Wikipedia.
"""
from __future__ import annotations

import re
import uuid

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import ActionStep, CodeAgent, PlanningStep
from smolagents.utils import AgentError

AGENT = AgentMeta(
    id="smol-plan-review",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/blob/main/examples/plan_customization/plan_customization.py",
    description="Planning CodeAgent whose plans pass through an auto-reviewer step callback that approves, edits, or cancels and resumes.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="openinference",
)

MAX_PLAN_STEPS = 5
UNSAFE = re.compile(
    r"\b(bypass|exploit|crack|phish|malware|keylogger|dox+|stalk|weapon|explosive|steal|password dump|ddos)\w*",
    re.IGNORECASE,
)
REVIEW_CONSTRAINTS = (
    "\n\nReviewer constraints (added by the plan reviewer):\n"
    "- Use at most 3 searches and at most 2 page visits.\n"
    "- Cite the source URL for every factual claim in the final answer.\n"
    "- If sources disagree, say so instead of picking one silently."
)


class AutoReviewer:
    """Simulated human in the loop for the source's approve / modify / cancel menu."""

    def __init__(self) -> None:
        self.decisions: list[str] = []
        self.rejection: str | None = None

    def __call__(self, memory_step: PlanningStep, agent: CodeAgent) -> None:
        plan = memory_step.plan or ""
        unsafe = UNSAFE.search(plan)
        if unsafe and self.rejection is None:
            # Choice 3: cancel. The run is resumed below with reset=False.
            self.decisions.append(f"cancel: plan mentions '{unsafe.group(0)}'")
            self.rejection = (
                f"The plan reviewer rejected your plan because it involves '{unsafe.group(0)}'. "
                "Do not pursue that. Explain briefly what you can and cannot help with, and offer a safe, "
                "legitimate alternative answer to the user's underlying need."
            )
            agent.interrupt()
            return
        steps = re.findall(r"^\s*(?:\d+[.)]|[-*])\s+.+$", plan, flags=re.MULTILINE)
        needs_sources = "source" not in plan.lower() and "cite" not in plan.lower()
        if len(steps) > MAX_PLAN_STEPS or needs_sources:
            # Choice 2: modify the plan in memory.
            kept = steps[:MAX_PLAN_STEPS] if len(steps) > MAX_PLAN_STEPS else None
            new_plan = ("\n".join(kept) if kept else plan.strip()) + REVIEW_CONSTRAINTS
            memory_step.plan = new_plan
            self.decisions.append(f"modify: {len(steps)} steps -> {len(kept) if kept else len(steps)}, constraints added")
            return
        # Choice 1: approve.
        self.decisions.append("approve")


def log_action(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Second step callback: keep only the last 3 observations at full length to bound context."""
    steps = [s for s in agent.memory.steps if isinstance(s, ActionStep) and s.observations]
    for old in steps[:-3]:
        if len(old.observations) > 1500:
            old.observations = old.observations[:1500] + "\n...[trimmed by step callback]"


def web_tools():
    return [
        C.wrap(T.web_search, name="web_search", output="string", limit=4000),
        C.wrap(T.fetch_url, name="visit_webpage", output="string", limit=6000),
        C.wrap(T.wikipedia_search, output="string", limit=3000),
    ]


def setup() -> None:
    C.setup_tracing()


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    reviewer = AutoReviewer()
    agent = CodeAgent(
        model=C.make_model(AGENT.model),
        tools=web_tools(),
        planning_interval=3,
        step_callbacks={PlanningStep: reviewer, ActionStep: log_action},
        max_steps=6,
        verbosity_level=0,
    )
    try:
        return C.last_text(agent.run(task["input"]))
    except AgentError as e:
        if "interrupted" not in str(e).lower() or reviewer.rejection is None:
            raise
    # Resume with memory preserved, as in the source's reset=False demonstration.
    agent.planning_interval = None
    return C.last_text(agent.run(reviewer.rejection, reset=False, max_steps=2))
