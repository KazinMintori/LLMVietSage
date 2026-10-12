"""Passive preferences produced by the optional response-style policy."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResponsePreferences:
    locale: str
    language_requested: bool = False
    detail: str | None = None
    tone: str | None = None
    format_hint: str | None = None
