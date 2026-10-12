"""Provider results and application events kept only during a request."""
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProviderReply:
    content: str
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderChunk:
    content: str = ""
    done: bool = False
    usage: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    name: Literal["token", "done", "error"]
    payload: dict
