"""DSPy math reasoning with code execution (dspy.RLM, the DSPy 3.5 replacement for ProgramOfThought/CodeAct).

Derived from the DSPy "Tutorial: Math Reasoning" (docs/docs/tutorials/math), which builds
`dspy.ChainOfThought("question -> answer")` and recommends a code sandbox for harder reasoning, and from the
RLM migration guide (docs/docs/api/modules/RLM.md): `dspy.ProgramOfThought` and `dspy.CodeAct` were removed in
DSPy 3.5 and `dspy.RLM("question -> answer", tools=[...])` is the documented replacement.
The RLM writes Python in a Deno/Pyodide sandbox (no host env or files), can call `llm_query`, and SUBMITs.
A host-side `sql_query` tool (tools service, Chinook/Northwind) is exposed for data-backed word problems.
Bounded: max_iters=6, max_llm_calls=3. No optimization (MIPROv2) is run.
"""

import dspy
import dsh_common
from peira_harness import AgentMeta, tools

AGENT = AgentMeta(
    id="dspy-math-rlm",
    framework="dspy",
    source_url="https://github.com/stanfordnlp/dspy/blob/main/docs/docs/tutorials/math/index.ipynb",
    description="dspy.RLM math solver that reasons by writing and running Python in a sandboxed REPL, with an SQL tool for data questions.",
    complexity="simple",
    model="claude-haiku-5-5",
    tracing="openinference",
)


def sql_query(db: str, sql: str) -> dict:
    """Run a read-only SQL query on a sample SQLite database. db is 'chinook' (music store) or 'northwind' (trading company)."""
    return tools.sql_query(db=db, sql=sql, max_rows=50)


def sql_schema(db: str) -> dict:
    """Return the tables and columns of a sample SQLite database ('chinook' or 'northwind')."""
    return tools.sql_schema(db=db)


class SolveMath(dspy.Signature):
    """Solve the math problem. Use Python to compute exact values instead of mental arithmetic.
    If the problem needs data, query the sample databases with the SQL tools. If the problem is
    ill-posed or has no solution, say so in the answer and explain why."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="final answer, simplified, with a one-line justification")


def setup() -> None:
    dsh_common.setup_dspy(AGENT.model)
    # Pre-warm the Deno/Pyodide sandbox: the first start downloads Pyodide from npm into DENO_DIR (~8s).
    # Failing here is clearer than every task timing out.
    interp = dspy.PythonInterpreter()
    try:
        interp.execute("print(1 + 1)")
    finally:
        interp.shutdown()


def run(task: dict) -> str:
    tools.set_run_id(f"{AGENT.id}:{task['id']}")
    solver = dspy.RLM(SolveMath, max_iters=6, max_llm_calls=3, max_output_chars=4000, tools=[sql_query, sql_schema])
    return solver(question=task["input"]).answer
