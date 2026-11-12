"""Game builder crew, adapted from crewAI-examples/crews/game-builder-crew.

Shape kept from the source: senior engineer writes the game, QA engineer reviews it, chief QA engineer
(allow_delegation=True) evaluates it and may delegate fixes back to coworkers. Sequential process.
Adapted: every engineer can run code in the sandboxed python_exec tool. The sandbox has no stdin, so each
game must include a scripted, non-interactive demo that exercises the game logic.
"""
import crew_common as cc  # first: sets CrewAI env (telemetry off) before crewai is imported

from crewai import Crew, Process, Task

from peira_harness import AgentMeta
from peira_harness import tools as T

AGENT = AgentMeta(
    id="crew-game-builder",
    framework="crewai",
    source_url="https://github.com/crewAIInc/crewAI-examples/tree/main/crews/game-builder-crew",
    description="Engineer, QA, and chief-QA crew that writes a Python game, tests it in a sandbox, and delegates fixes.",
    complexity="medium",
    model=cc.HAIKU,
    tracing="openinference",
)

SANDBOX_NOTE = (
    "The python_exec sandbox has no stdin, no network, and no display: never call input(), pygame, curses or "
    "tkinter in code you run. Keep the game logic in functions or classes and add a scripted demo under "
    "`if __name__ == '__main__':` that plays a few turns automatically and prints the result."
)


def setup() -> None:
    cc.setup_tracing()


def build_crew(run_id: str) -> Crew:
    engineer = cc.mk_agent(
        role="Senior Software Engineer",
        goal="Create software as needed",
        backstory="You are a Senior Software Engineer at a leading tech think tank. Your expertise in programming in "
        "python. and do your best to produce perfect code. " + SANDBOX_NOTE,
        tools=cc.make_tools([T.python_exec], run_id, max_chars=4000),
        max_iter=4,
    )
    qa = cc.mk_agent(
        role="Software Quality Control Engineer",
        goal="Create Perfect code, by analyzing the code that is given for errors",
        backstory="You are a software engineer that specializes in checking code for errors. You have an eye for detail "
        "and a knack for finding hidden bugs. You check for missing imports, variable declarations, mismatched brackets "
        "and syntax errors. You also check for security vulnerabilities, and logic errors. " + SANDBOX_NOTE,
        tools=cc.make_tools([T.python_exec], run_id, max_chars=4000),
        max_iter=3,
    )
    chief_qa = cc.mk_agent(
        role="Chief Software Quality Control Engineer",
        goal="Ensure that the code does the job that it is supposed to do",
        backstory="You feel that programmers always do only half the job, so you are super dedicate to make high "
        "quality code. You can delegate a focused fix to the Senior Software Engineer when something is broken. "
        + SANDBOX_NOTE,
        tools=cc.make_tools([T.python_exec], run_id, max_chars=4000),
        allow_delegation=True,
        max_iter=4,
    )
    instructions = "Instructions\n------------\n{request}\n\n" + SANDBOX_NOTE
    code_task = Task(
        description="You will create a game using python, these are the instructions:\n\n" + instructions +
        "\n\nRun your code once with python_exec before answering.",
        expected_output="Your Final answer must be the full python code, only the python code and nothing else.",
        agent=engineer,
    )
    review_task = Task(
        description="You will create a game using python, these are the instructions:\n\n" + instructions +
        "\n\nUsing the code you got, check for errors. Check for logic errors, syntax errors, missing imports, variable "
        "declarations, mismatched brackets, and security vulnerabilities.",
        expected_output="Your Final answer must be the full python code, only the python code and nothing else.",
        agent=qa,
        context=[code_task],
    )
    evaluate_task = Task(
        description="You are helping create a game using python, these are the instructions:\n\n" + instructions +
        "\n\nYou will look over the code to insure that it is complete and does the job that it is supposed to do. "
        "Run it. If the instructions are impossible or unsafe, explain that briefly in a comment at the top instead.",
        expected_output="Your Final answer must be the full python code, only the python code and nothing else.",
        agent=chief_qa,
        context=[review_task],
    )
    return Crew(
        agents=[engineer, qa, chief_qa],
        tasks=[code_task, review_task, evaluate_task],
        process=Process.sequential,
        verbose=False,
    )


def run(task: dict) -> str:
    crew = build_crew(cc.run_id_for(AGENT.id, task))
    return cc.result_text(crew.kickoff(inputs={"request": task["input"]}))
