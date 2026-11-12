"""GAIA-style generalist, adapted from the smolagents benchmark runner and run_gaia.py.

Source shape:
- examples/smolagents_benchmark/run.py, action_type "tool-calling": one ToolCallingAgent with
  [GoogleSearchTool, VisitWebpageTool, PythonInterpreterTool(authorized_imports=["numpy", "sympy"])].
- examples/open_deep_research/run_gaia.py: planning_interval, a `final_answer_checks` hook, and after
  the run the `prepare_response` reformulator (scripts/reformulator.py) that turns the transcript into
  a strict "FINAL ANSWER: ..." with GAIA formatting rules.
Adaptation: Serper search -> tools-service web search; VisitWebpageTool -> tools-service fetch;
PythonInterpreterTool -> the tools-service python sandbox (code never runs in the agent pod);
Wikipedia and arXiv added as free research tools. The reformulator is kept verbatim in spirit.
"""
from __future__ import annotations

import copy
import uuid

import smol_common as C
from peira_harness import AgentMeta
from peira_harness import tools as T
from smolagents import Tool, ToolCallingAgent
from smolagents.models import MessageRole, Model

AGENT = AgentMeta(
    id="smol-gaia-generalist",
    framework="smolagents",
    source_url="https://github.com/huggingface/smolagents/blob/main/examples/open_deep_research/run_gaia.py",
    description="GAIA-style generalist ToolCallingAgent with planning, sandboxed Python, research tools, and a transcript reformulator.",
    complexity="complex",
    model="claude-sonnet-5-5",
    tracing="openinference",
)


class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = (
        "This is a tool that evaluates python code. It can be used to perform calculations. "
        "The code runs in an isolated sandbox with numpy, pandas, scipy and sympy installed and no network. "
        "Each call is independent: no variables persist. You must print() what you want to see."
    )
    inputs = {"code": {"type": "string", "description": "The python code to run in interpreter"}}
    output_type = "string"

    def forward(self, code: str) -> str:
        res = T.python_exec(code, timeout_s=20)
        if "error" in res:
            return f"Sandbox error: {res['error']}"
        out = f"Stdout:\n{res.get('stdout', '')}"
        if res.get("status") != "ok":
            out += f"\nStatus: {res.get('status')}\nStderr:\n{res.get('stderr', '')}"
        return out[:6000]


def _tools() -> list[Tool]:
    return [
        C.wrap(T.web_search, output="string", limit=4000),
        C.wrap(T.fetch_url, name="visit_webpage", output="string", limit=8000),
        C.wrap(T.wikipedia_search, output="string", limit=3000),
        C.wrap(T.wikipedia_page, output="string", limit=8000),
        C.wrap(T.arxiv_search, output="string", limit=5000),
        PythonInterpreterTool(),
    ]


def answer_is_not_empty(final_answer, memory, agent=None) -> bool:
    """run_gaia.py hooks `final_answer_checks`; here it rejects empty or placeholder answers."""
    text = str(final_answer).strip()
    if not text or text.lower() in {"none", "n/a", "todo", "answer"}:
        raise ValueError("The final answer is empty. Give your best supported answer, or explain why it cannot be determined.")
    return True


def prepare_response(original_task: str, inner_messages, reformulation_model: Model) -> str:
    """Port of open_deep_research/scripts/reformulator.py."""
    messages = [
        {
            "role": MessageRole.SYSTEM,
            "content": [
                {
                    "type": "text",
                    "text": f"""Earlier you were asked the following:

{original_task}

Your team then worked diligently to address that request. Read below a transcript of that conversation:""",
                }
            ],
        }
    ]
    for message in inner_messages:
        # Skip the agent's own system prompt: it is long and not part of the conversation.
        if not message.content or message.role == MessageRole.SYSTEM:
            continue
        message = copy.deepcopy(message)
        message.role = MessageRole.USER
        message.tool_calls = None
        messages.append(message)
    messages.append(
        {
            "role": MessageRole.USER,
            "content": [
                {
                    "type": "text",
                    "text": f"""
Read the above conversation and output a FINAL ANSWER to the question. The question is repeated here for convenience:

{original_task}

To output the final answer, use the following template: FINAL ANSWER: [YOUR FINAL ANSWER]
Your FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
ADDITIONALLY, your FINAL ANSWER MUST adhere to any formatting instructions specified in the original question (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.)
If you are asked for a number, express it numerically (i.e., with digits rather than words), don't use commas, and DO NOT INCLUDE UNITS such as $ or USD or percent signs unless specified otherwise.
If you are asked for a string, don't use articles or abbreviations (e.g. for cities), unless specified otherwise. Don't output any final sentence punctuation such as '.', '!', or '?'.
If you are asked for a comma separated list, apply the above rules depending on whether the elements are numbers or strings.
If the question is not a short-answer question (for example it asks for an explanation), keep the explanation but make it concise.
If you are unable to determine the final answer, output 'FINAL ANSWER: Unable to determine' followed by one sentence on why.
""",
                }
            ],
        }
    )
    response = reformulation_model(messages).content
    return response.split("FINAL ANSWER: ")[-1].strip()


def setup() -> None:
    C.setup_tracing()


def run(task: dict) -> str:
    T.set_run_id(f"{AGENT.id}:{task['id']}:{uuid.uuid4().hex[:8]}")
    model = C.make_model(AGENT.model)
    agent = ToolCallingAgent(
        tools=_tools(),
        model=model,
        max_steps=8,
        planning_interval=4,
        verbosity_level=0,
        final_answer_checks=[answer_is_not_empty],
        max_tool_threads=4,
    )
    question = task["input"]
    augmented = (
        "You have one question to answer. It is paramount that you provide a correct answer. "
        "Give it all you can: I know for a fact that you have access to all the relevant tools to solve it "
        "and find the correct answer (the answer does exist). Failure or 'I cannot answer' or 'None found' "
        "will not be tolerated, success will be rewarded. Run verification steps if that's needed, you must "
        "make sure you find the correct answer! However, if the question is impossible, contradictory, or "
        "asks you to do something unsafe, say so clearly instead.\nHere is the task:\n" + question
    )
    agent.run(augmented)
    return prepare_response(question, agent.write_memory_to_messages(), reformulation_model=model)
