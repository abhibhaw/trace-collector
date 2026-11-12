"""Reflexion (Shinn et al., 2023) for Python programming tasks.

Adapted from noahshinn/reflexion programming_runs/reflexion.py and generators/py_generate.py:
generate internal unit tests -> implement -> execute tests -> on failure, write a verbal self-reflection
-> re-implement using the previous implementation, the test feedback, and the reflection memory ->
repeat until the tests pass or max_iters is reached. Code runs in the tools service's sandboxed
python_exec instead of a local executor. Every attempt is a chain span; test runs are tool spans.
"""
from __future__ import annotations

import json
import re

from peira_harness import AgentMeta, llm
from peira_harness import tools as T
from raw_common import ModelRefused, Toolbox, anthropic_client, begin_task, extract_xml, instrument_anthropic, phase, refusal_text, text_of

AGENT = AgentMeta(
    id="raw-reflexion-coder",
    framework="raw-anthropic",
    source_url="https://github.com/noahshinn/reflexion/blob/main/programming_runs/reflexion.py",
    description="Reflexion coding loop: self-generated unit tests, sandboxed execution, verbal self-reflection memory, up to 3 implementation attempts.",
    complexity="complex",
    model=llm.SONNET,
    tracing="manual",
)

MAX_ITERS = 3
MAX_TESTS = 6

# --- prompts from programming_runs/generators/py_generate.py (chat variants) ---
PY_SIMPLE_CHAT_INSTRUCTION = (
    "You are an AI that only responds with python code, NOT ENGLISH. You will be given a function signature and its "
    "docstring by the user. Write your full implementation (restate the function signature). Use only the standard "
    "library unless the task says otherwise."
)
PY_REFLEXION_CHAT_INSTRUCTION = (
    "You are an AI Python assistant. You will be given your past function implementation, a series of unit tests, "
    "and a hint to change the implementation appropriately. Write your full implementation (restate the function signature)."
)
PY_SELF_REFLECTION_CHAT_INSTRUCTION = (
    "You are a Python programming assistant. You will be given a function implementation and a series of unit tests. "
    "Your goal is to write a few sentences to explain why your implementation is wrong as indicated by the tests. "
    "You will need this as a hint when you try again later. Only provide the few sentence description in your answer, "
    "not the implementation. If a test itself looks wrong for the specification, say so."
)
PY_TEST_GENERATION_CHAT_INSTRUCTION = (
    "You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the "
    "signature and docstring."
)
USE_PYTHON_CODEBLOCK_INSTRUCTION = "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"

SPEC_PROMPT = f"""Turn the user's programming request into a precise Python function specification and unit tests.
If the request does not say, choose a sensible function name and argument types.
If the request is not a programming task, is impossible to implement as a pure function, or asks for something
harmful, leave <signature> empty and explain in <problem>.

Return:
<signature>
def name(args) -> type:
    \"\"\"Docstring with the exact behaviour, edge cases, and 1-2 examples.\"\"\"
</signature>
<tests>
assert name(...) == ...
(at most {MAX_TESTS} single-line asserts, covering normal and edge cases; each assert on one line)
</tests>
<problem>empty, or why this cannot be done</problem>

User request:
"""

TEST_HARNESS = '''
import json, traceback
__results = []
for __src in {tests!r}:
    try:
        exec(__src, globals())
        __results.append({{"test": __src, "passed": True}})
    except AssertionError:
        try:
            __lhs = __src.split("assert", 1)[1].rsplit("==", 1)[0].strip()
            __got = repr(eval(__lhs, globals()))
        except Exception as __e:
            __got = f"{{type(__e).__name__}}: {{__e}}"
        __results.append({{"test": __src, "passed": False, "output": __got[:300]}})
    except Exception as __e:
        __results.append({{"test": __src, "passed": False, "output": f"{{type(__e).__name__}}: {{__e}}"[:300]}})
print("__RESULTS__" + json.dumps(__results))
'''

TOOLS = Toolbox([T.python_exec])
_client = None


def setup() -> None:
    instrument_anthropic()


def chat(system: str, user: str) -> str:
    resp = _client.messages.create(
        model=AGENT.model, max_tokens=llm.MAX_TOKENS, system=system, messages=[{"role": "user", "content": user}]
    )
    if (refused := refusal_text(resp)) is not None:
        raise ModelRefused(refused)
    return text_of(resp)


def parse_code_block(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    return (max(m, key=len) if m else text).strip()


def execute(impl: str, tests: list[str]) -> tuple[bool, str, list[dict]]:
    """Run the implementation plus every test in the sandbox; return (all passed, feedback text, results)."""
    code = impl + "\n\n" + TEST_HARNESS.format(tests=tests)
    raw, err = TOOLS.call("python_exec", {"code": code, "timeout_s": 20})
    results: list[dict] = []
    if not err:
        try:
            out = json.loads(raw)
        except json.JSONDecodeError:
            out = {"stdout": raw}
        stdout = out.get("stdout", "") if isinstance(out, dict) else str(out)
        marker = stdout.rfind("__RESULTS__")
        if marker >= 0:
            try:
                results = json.loads(stdout[marker + len("__RESULTS__"):].strip().splitlines()[0])
            except (json.JSONDecodeError, IndexError):
                results = []
        if not results:
            stderr = out.get("stderr", "") if isinstance(out, dict) else ""
            return False, f"The implementation did not run. status={out.get('status') if isinstance(out, dict) else '?'}\n{stderr[-1500:]}", []
    else:
        return False, f"Sandbox error: {raw}", []
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    feedback = "Tested passed:\n" + "\n".join(r["test"] for r in passed)
    feedback += "\n\nTests failed:\n" + "\n".join(f"{r['test']} # output: {r.get('output')}" for r in failed)
    return not failed, feedback, results


def run(task: dict) -> str:
    begin_task(AGENT.id, task)
    global _client
    if _client is None:
        _client = anthropic_client()
    try:
        return _run(task["input"])
    except ModelRefused as e:
        return str(e)


def _run(request: str) -> str:

    with phase("spec_and_internal_tests", "chain", input=request) as sp:
        out = chat(PY_TEST_GENERATION_CHAT_INSTRUCTION, SPEC_PROMPT + request)
        signature = extract_xml(out, "signature")
        tests = [ln.strip() for ln in extract_xml(out, "tests").splitlines() if ln.strip().startswith("assert")][:MAX_TESTS]
        problem = extract_xml(out, "problem")
        sp.update(output={"signature": signature, "tests": tests, "problem": problem})
    if not signature:
        return f"I can't turn this into a function to implement and test. {problem}".strip()
    if not tests:
        raise RuntimeError("test generation produced no asserts")

    reflections: list[str] = []
    impl, feedback, is_passing, results = "", "", False, []
    for it in range(1, MAX_ITERS + 1):
        with phase(f"attempt-{it}", "chain", metadata={"strategy": "simple" if it == 1 else "reflexion"}) as att:
            if it == 1:
                with phase("implement", "chain"):
                    impl = parse_code_block(chat(PY_SIMPLE_CHAT_INSTRUCTION + "\n" + USE_PYTHON_CODEBLOCK_INSTRUCTION, signature))
            else:
                with phase("self_reflection", "evaluator") as rf:
                    reflection = chat(
                        PY_SELF_REFLECTION_CHAT_INSTRUCTION,
                        f"[function impl]:\n```python\n{impl}\n```\n\n[unit test results]:\n{feedback}\n\n[self-reflection]:",
                    )
                    reflections.append(reflection)
                    rf.update(output=reflection)
                with phase("implement", "chain"):
                    memory = "\n".join(f"- attempt {i + 1}: {r}" for i, r in enumerate(reflections))
                    impl = parse_code_block(chat(
                        PY_REFLEXION_CHAT_INSTRUCTION + "\n" + USE_PYTHON_CODEBLOCK_INSTRUCTION,
                        f"[previous impl]:\n```python\n{impl}\n```\n\n[unit test results from previous impl]:\n{feedback}\n\n"
                        f"[reflections so far]:\n{memory}\n\n[improved impl]:\n{signature}",
                    ))
            is_passing, feedback, results = execute(impl, tests)
            att.update(output={"passed": sum(r["passed"] for r in results), "total": len(tests), "all_passed": is_passing})
        if is_passing:
            break

    n_pass = sum(r["passed"] for r in results)
    header = (f"All {len(tests)} self-generated tests pass after {it} attempt(s)." if is_passing
              else f"{n_pass}/{len(tests)} self-generated tests pass after {MAX_ITERS} attempts; the remaining failures are listed below.")
    parts = [header, f"```python\n{impl}\n```", "Tests:\n" + "\n".join(tests)]
    if not is_passing:
        parts.append(feedback)
    if reflections:
        parts.append("Reflections:\n" + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reflections)))
    return "\n\n".join(parts)
