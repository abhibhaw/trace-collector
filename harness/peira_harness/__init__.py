"""Shared harness for the peira trace-collection experiment. See CONTRACT.md."""
from dataclasses import dataclass
from typing import Literal

Complexity = Literal["simple", "medium", "complex"]
TracingMode = Literal["langchain", "openinference", "native-otel", "manual"]


@dataclass(frozen=True)
class AgentMeta:
    id: str
    framework: str
    source_url: str
    description: str
    complexity: Complexity
    model: str = "claude-haiku-5-5"
    tracing: TracingMode = "openinference"

    def __post_init__(self) -> None:
        if self.model not in ("claude-haiku-5-5", "claude-sonnet-5-5"):
            raise ValueError(f"{self.id}: model must be claude-haiku-5-5 or claude-sonnet-5-5")
        if len(self.id) > 40 or self.id != self.id.lower():
            raise ValueError(f"{self.id}: id must be lowercase and at most 40 chars")


__all__ = ["AgentMeta"]
