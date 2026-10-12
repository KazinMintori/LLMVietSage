"""Serialize application events as SSE without making generation decisions."""
import json
from collections.abc import AsyncGenerator
from typing import Protocol


class SSEEvent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def payload(self) -> dict: ...


def encode_sse_event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def encode_sse_stream(events: AsyncGenerator[SSEEvent, None]) -> AsyncGenerator[str, None]:
    try:
        async for event in events:
            yield encode_sse_event(event.name, event.payload)
    finally:
        await events.aclose()
