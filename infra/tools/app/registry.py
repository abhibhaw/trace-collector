"""Tool registry. Each tool is an async function; pydantic validates its arguments."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import validate_call

TOOLS: dict[str, dict[str, Any]] = {}


class ToolError(Exception):
    """A readable error for the model (not found, invalid input, upstream unavailable)."""


def tool(group: str, write: bool = False) -> Callable:
    def deco(fn: Callable) -> Callable:
        assert inspect.iscoroutinefunction(fn), fn.__name__
        TOOLS[fn.__name__] = {"fn": validate_call(fn), "group": group, "write": write,
                              "doc": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else "",
                              "params": [p for p in inspect.signature(fn).parameters]}
        return fn
    return deco
