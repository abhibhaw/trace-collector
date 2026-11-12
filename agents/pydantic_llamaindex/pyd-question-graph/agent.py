"""Quiz state machine, adapted from PydanticAI's question_graph example (pydantic-graph).

Shape kept from the source: a pydantic-graph with BaseNode states Ask -> Answer -> Evaluate,
and Evaluate -> Reprimand -> Ask on a wrong answer, with per-agent message history in the
graph state and a structured EvaluationOutput. Adaptations:
- The human `input()` in Answer is replaced by a student agent that answers with web tools.
- Ask and Evaluate agents ground the question and the grading in Wikipedia.
- The loop is capped at 3 rounds.
- The graph is built with the pydantic-graph 2.x GraphBuilder (the v1 Graph(nodes=...) API was removed).
- pydantic-graph emits node spans only through logfire, so each node opens a plain OTel span.
"""

from dataclasses import dataclass, field

from opentelemetry import trace
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage, UsageLimits, format_as_xml
from pydantic_ai.usage import RunUsage
from pydantic_graph import BaseNode, End, GraphBuilder, GraphRunContext, StepContext

import pl_common as pc
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="pyd-question-graph",
    framework="pydantic-ai",
    source_url="https://github.com/pydantic/pydantic-ai/blob/v1.0.0/examples/pydantic_ai_examples/question_graph.py",
    description="pydantic-graph state machine: ask a grounded question, a student agent answers, an evaluator grades, reprimand loops.",
    complexity="medium",
    model="claude-haiku-5-5",
    tracing="native-otel",
)

MAX_ROUNDS = 3
LIMITS = UsageLimits(request_limit=24)
_tracer = trace.get_tracer("pyd-question-graph")

wiki_search = pc.safe_tool(tools.wikipedia_search)
wiki_page = pc.safe_tool(tools.wikipedia_page)
web_search = pc.safe_tool(tools.web_search)

ask_agent = Agent(
    pc.pyd_model(),
    name="ask_agent",
    output_type=str,
    tools=[wiki_search, wiki_page],
    instructions=(
        "You are a quiz master. Ask one challenging question with a single correct, short answer (a specific name, "
        "date, number, or place) about the user's requested topic. Avoid textbook-famous facts. "
        "Check the fact in Wikipedia first (at most 2 tool calls). Output only the question text. "
        "If earlier questions were answered wrongly, ask a different, slightly easier question. "
        "If the request is not a quiz topic or is unsafe, ask a general-knowledge question related to it instead."
    ),
)

student_agent = Agent(
    pc.pyd_model(),
    name="student_agent",
    output_type=str,
    tools=[web_search, wiki_search],
    instructions=(
        "You are a student taking a quiz under time pressure. Answer the question in one short sentence. "
        "Answer from memory when you think you know; look things up at most once. Do not ask questions back."
    ),
)


class EvaluationOutput(BaseModel, use_attribute_docstrings=True):
    correct: bool
    """Whether the answer is correct."""
    comment: str
    """Comment on the answer, reprimand the user if the answer is wrong."""


evaluate_agent = Agent(
    pc.pyd_model(),
    name="evaluate_agent",
    output_type=EvaluationOutput,
    tools=[wiki_search],
    instructions="Given a question and answer, evaluate if the answer is correct. Verify with one search if you are unsure.",
)


@dataclass
class QuestionState:
    topic: str
    usage: RunUsage = field(default_factory=RunUsage)
    question: str | None = None
    rounds: int = 0
    transcript: list[dict] = field(default_factory=list)
    ask_agent_messages: list[ModelMessage] = field(default_factory=list)
    evaluate_agent_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Ask(BaseNode[QuestionState]):
    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Answer":
        with _tracer.start_as_current_span("node Ask"):
            ctx.state.rounds += 1
            prompt = (
                f"Topic request: {ctx.state.topic}\nAsk question #{ctx.state.rounds}."
                if not ctx.state.ask_agent_messages
                else f"Ask question #{ctx.state.rounds} on the same topic."
            )
            result = await ask_agent.run(
                prompt, message_history=ctx.state.ask_agent_messages, usage=ctx.state.usage, usage_limits=LIMITS
            )
            ctx.state.ask_agent_messages += result.new_messages()
            ctx.state.question = result.output
            return Answer(result.output)


@dataclass
class Answer(BaseNode[QuestionState]):
    question: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> "Evaluate":
        with _tracer.start_as_current_span("node Answer"):
            result = await student_agent.run(self.question, usage=ctx.state.usage, usage_limits=LIMITS)
            return Evaluate(result.output)


@dataclass
class Evaluate(BaseNode[QuestionState, None, str]):
    answer: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> "End[str] | Reprimand":
        with _tracer.start_as_current_span("node Evaluate"):
            assert ctx.state.question is not None
            result = await evaluate_agent.run(
                format_as_xml({"question": ctx.state.question, "answer": self.answer}),
                message_history=ctx.state.evaluate_agent_messages,
                usage=ctx.state.usage,
                usage_limits=LIMITS,
            )
            ctx.state.evaluate_agent_messages += result.new_messages()
            ctx.state.transcript.append(
                {"round": ctx.state.rounds, "question": ctx.state.question, "answer": self.answer,
                 "correct": result.output.correct, "comment": result.output.comment}
            )
            if result.output.correct:
                return End(result.output.comment)
            if ctx.state.rounds >= MAX_ROUNDS:
                return End(f"Stopped after {MAX_ROUNDS} rounds without a correct answer. Last comment: {result.output.comment}")
            return Reprimand(result.output.comment)


@dataclass
class Reprimand(BaseNode[QuestionState]):
    comment: str

    async def run(self, ctx: GraphRunContext[QuestionState]) -> Ask:
        with _tracer.start_as_current_span("node Reprimand"):
            ctx.state.question = None
            return Ask()


g = GraphBuilder(state_type=QuestionState, output_type=str)


@g.step
async def start(ctx: StepContext[QuestionState, None, None]) -> Ask:
    return Ask()


g.add(
    g.node(Ask),
    g.node(Answer),
    g.node(Evaluate),
    g.node(Reprimand),
    g.edge_from(g.start_node).to(start),
)
question_graph = g.build()


def setup() -> None:
    pc.pyd_setup()


async def arun(task: dict) -> str:
    pc.begin_task(AGENT.id, task)
    state = QuestionState(topic=task["input"])
    with _tracer.start_as_current_span("question_graph"):
        final = await question_graph.run(state=state)
    lines = [f"Final: {final}", f"Rounds: {state.rounds}"]
    for t in state.transcript:
        mark = "correct" if t["correct"] else "wrong"
        lines.append(f"Q{t['round']}: {t['question']}\nA: {t['answer']}\nEvaluation ({mark}): {t['comment']}")
    return "\n\n".join(lines)
