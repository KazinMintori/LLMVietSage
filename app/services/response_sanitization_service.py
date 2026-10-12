"""Policies for visible text and reasoning-tag removal."""
import json
import re
from collections.abc import AsyncGenerator

from app.domain.errors import InvalidModelResponseError


def _partial_tag_suffix(text: str, tags: tuple[str, ...]) -> int:
    """Length of a suffix that may be the beginning of a split XML tag."""
    return max(
        (
            size
            for tag in tags
            for size in range(1, min(len(tag), len(text) + 1))
            if text.endswith(tag[:size])
        ),
        default=0,
    )


async def strip_thinking_stream(
    chunks: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Remove ``<think>`` blocks without buffering the visible answer."""
    open_tag = "<think>"
    close_tag = "</think>"
    tags = (open_tag, close_tag)
    buffer = ""
    inside_thinking = False

    try:
        async for chunk in chunks:
            if not chunk:
                continue
            buffer += chunk

            while buffer:
                if inside_thinking:
                    end = buffer.find(close_tag)
                    if end < 0:
                        keep = _partial_tag_suffix(buffer, (close_tag,))
                        buffer = buffer[-keep:] if keep else ""
                        break
                    buffer = buffer[end + len(close_tag):]
                    inside_thinking = False
                    continue

                open_at = buffer.find(open_tag)
                close_at = buffer.find(close_tag)
                positions = [pos for pos in (open_at, close_at) if pos >= 0]
                if positions:
                    marker_at = min(positions)
                    visible = buffer[:marker_at]
                    if visible:
                        yield visible
                    if marker_at == open_at:
                        buffer = buffer[marker_at + len(open_tag):]
                        inside_thinking = True
                    else:
                        buffer = buffer[marker_at + len(close_tag):]
                    continue

                keep = _partial_tag_suffix(buffer, tags)
                visible = buffer[:-keep] if keep else buffer
                buffer = buffer[-keep:] if keep else ""
                if visible:
                    yield visible
                break

        # Remaining visible buffer is only an unresolved tag prefix; do not leak
        # a truncated "<think>" marker when the provider stops mid-tag.
    finally:
        close = getattr(chunks, "aclose", None)
        if close is not None:
            await close()


def strip_thinking(text: str) -> str:
    """Remove reasoning tags and their contents from a completed reply."""
    if "</think>" in text and "<think>" not in text:
        text = text.rsplit("</think>", 1)[-1]
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    text = re.sub(r"<think>[\s\S]*", "", text)
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _reject_json_constant(value: str):
    raise InvalidModelResponseError("Ollama returned a non-finite JSON value.")


def parse_structured_reply(text: str) -> dict:
    """Validate a JSON object; full caller JSON Schema validation is separate."""
    try:
        parsed = json.loads(strip_thinking(text), parse_constant=_reject_json_constant)
    except (TypeError, ValueError) as exc:
        raise InvalidModelResponseError("Ollama returned invalid structured JSON.") from exc
    if not isinstance(parsed, dict):
        raise InvalidModelResponseError("Ollama structured response must be a JSON object.")
    return parsed
