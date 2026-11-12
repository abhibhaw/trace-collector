"""Shared helpers for the smolagents family: proxy model, tracing, and tool adapters.

The runner puts the family directory on sys.path, so every agent can `import smol_common`.
"""
from __future__ import annotations

import inspect
import json
import re
import types
import typing
from typing import Any, Callable

from peira_harness import llm, tracing
from smolagents import LiteLLMModel, Tool

MAX_TOKENS = 4096


class ProxyModel(LiteLLMModel):
    """LiteLLMModel pointed at the LiteLLM proxy (Anthropic format).

    smolagents sends `tool_choice="required"` on every ToolCallingAgent step.
    claude-sonnet-5-5 rejects forced tool choice, so we downgrade it to "auto"
    unless `force_tool_choice=True`. Only `_prepare_completion_kwargs` is overridden,
    so the OpenInference wrapper on `LiteLLMModel.generate` still applies.
    """

    def __init__(self, model: str = llm.HAIKU, force_tool_choice: bool = False, **kw):
        kw.setdefault("max_tokens", MAX_TOKENS)
        kw.setdefault("num_retries", 4)
        self.force_tool_choice = force_tool_choice
        super().__init__(
            model_id=f"anthropic/{model}",
            api_base=llm.anthropic_base_url(),
            api_key=llm.proxy_key(),
            **kw,
        )

    def _prepare_completion_kwargs(self, *args, **kwargs) -> dict[str, Any]:
        out = super()._prepare_completion_kwargs(*args, **kwargs)
        if out.get("tool_choice") == "required" and not self.force_tool_choice:
            out["tool_choice"] = "auto"
        # Model rules: never send sampling params.
        for k in ("temperature", "top_p", "top_k"):
            out.pop(k, None)
        return out


def make_model(model: str = llm.HAIKU, **kw) -> ProxyModel:
    return ProxyModel(model=model, **kw)


def setup_tracing() -> None:
    """Instrument smolagents once per process. This is the only LLM span source."""
    from openinference.instrumentation.smolagents import SmolagentsInstrumentor

    _name_tool_spans()
    tracing.instrument(SmolagentsInstrumentor())


_NAMED: dict[tuple[type, str], type] = {}


def _name_tool_spans() -> None:
    """OpenInference names tool spans after the tool's class (`SimpleTool` for every @tool function).

    Patch agent tool setup so each tool instance gets a thin subclass named after `tool.name`,
    which makes tool spans read `web_search`, `sql_engine`, `final_answer`, ...
    """
    from smolagents.agents import MultiStepAgent

    if getattr(MultiStepAgent, "_peira_named_tools", False):
        return
    original = MultiStepAgent._setup_tools

    def _setup_tools(self, tools, add_base_tools):
        original(self, tools, add_base_tools)
        for t in self.tools.values():
            cls = type(t)
            if cls.__name__ == t.name:
                continue
            key = (cls, t.name)
            if key not in _NAMED:
                _NAMED[key] = type(t.name, (cls,), {"__module__": cls.__module__, "skip_forward_signature_validation": True})
            t.__class__ = _NAMED[key]

    MultiStepAgent._setup_tools = _setup_tools
    MultiStepAgent._peira_named_tools = True


# --------------------------------------------------------------------------- tool adapter

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def _json_type(ann: Any) -> tuple[str, bool]:
    """Map a Python annotation to (smolagents type, nullable)."""
    if ann is inspect.Parameter.empty:
        return "string", False
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        nullable = len(args) < len(typing.get_args(ann))
        t, _ = _json_type(args[0]) if args else ("string", False)
        return t, nullable
    if origin is typing.Literal:
        v = typing.get_args(ann)[0]
        return _TYPE_MAP.get(type(v), "string"), False
    if origin is not None:
        return _TYPE_MAP.get(origin, "object"), False
    return _TYPE_MAP.get(ann, "any"), False


def _parse_doc(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {param: description})."""
    doc = inspect.cleandoc(doc or "")
    parts = re.split(r"\n\s*(Args|Arguments|Parameters):\s*\n", doc, maxsplit=1)
    summary = parts[0].strip()
    params: dict[str, str] = {}
    if len(parts) == 3:
        body = re.split(r"\n\s*(Returns|Raises|Yields|Example|Examples):", parts[2])[0]
        cur = None
        for line in body.splitlines():
            m = re.match(r"\s*(\w+)\s*(\([^)]*\))?\s*:\s*(.*)", line)
            if m and (cur is None or len(line) - len(line.lstrip()) <= 4):
                cur = m.group(1)
                params[cur] = m.group(3).strip()
            elif cur and line.strip():
                params[cur] += " " + line.strip()
    return summary, params


def _stringify(result: Any, limit: int) -> str:
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(result)
    if limit and len(text) > limit:
        text = text[:limit] + f"\n...[truncated {len(text) - limit} chars]"
    return text


def wrap(
    fn: Callable,
    name: str | None = None,
    description: str | None = None,
    output: str = "string",
    limit: int = 6000,
) -> Tool:
    """Wrap a plain typed Python function as a smolagents Tool.

    output="string": JSON-encode the result (best for ToolCallingAgent).
    output="object": return the raw Python value (best for CodeAgent, which keeps it as a variable).
    """
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn)
    summary, pdocs = _parse_doc(fn.__doc__ or "")
    inputs: dict[str, dict] = {}
    for pname, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        t, nullable = _json_type(hints.get(pname, p.annotation))
        spec = {"type": t, "description": pdocs.get(pname) or pname.replace("_", " ")}
        if nullable or p.default is not inspect.Parameter.empty:
            spec["nullable"] = True
        inputs[pname] = spec

    tool_name = name or fn.__name__
    tool_desc = description or summary or tool_name
    params = list(inputs)

    def forward(self, *args, **kwargs):
        # Models often send explicit nulls for optional args: fall back to the function default.
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if not (v is None and k in sig.parameters and sig.parameters[k].default is not inspect.Parameter.empty)
        }
        result = fn(*args, **kwargs)
        return result if output == "object" else _stringify(result, limit)

    # smolagents validates that forward's signature matches `inputs` exactly.
    forward.__signature__ = inspect.Signature(
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        + [
            inspect.Parameter(
                n,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=(None if inputs[n].get("nullable") else inspect.Parameter.empty),
            )
            for n in params
        ]
    )
    cls = type(
        f"{tool_name.title().replace('_', '')}Tool",
        (Tool,),
        {
            "name": tool_name,
            "description": tool_desc,
            "inputs": inputs,
            "output_type": output if output != "string" else "string",
            "forward": forward,
            "skip_forward_signature_validation": True,
        },
    )
    return cls()


def last_text(result: Any) -> str:
    """Normalise an agent result (or RunResult) into the string the runner records."""
    out = getattr(result, "output", result)
    if isinstance(out, str):
        return out
    try:
        return json.dumps(out, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(out)
