"""Model-generation policy over an injected provider, without HTTP responses."""
from collections.abc import AsyncGenerator
from contextvars import ContextVar

from app.domain.models.provider import ModelProvider
from app.domain.errors import InvalidModelResponseError
from app.services.response_sanitization_service import (
    parse_structured_reply, strip_thinking, strip_thinking_stream,
)
from config.generation import OUTPUT_LIMIT_NOTICE

_LAST_CHAT_USAGE: ContextVar[dict | None] = ContextVar("vietsage_last_chat_usage", default=None)


def current_chat_usage() -> dict:
    """Copy counters for this async request; never retain caller messages."""
    return dict(_LAST_CHAT_USAGE.get() or {})


class LLMService:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    async def startup(self) -> None:
        await self.provider.startup()

    async def shutdown(self) -> None:
        await self.provider.shutdown()

    async def warmup(self) -> None:
        await self.provider.chat([{"role": "user", "content": "Chỉ trả lời đúng một từ: OK"}])

    async def chat(self, messages: list[dict], *, num_predict=None, num_ctx=None) -> str:
        _LAST_CHAT_USAGE.set(None)
        result = await self.provider.chat(messages, num_predict=num_predict, num_ctx=num_ctx)
        _LAST_CHAT_USAGE.set(dict(result.usage))
        answer = result.content
        if result.usage.get("done_reason") == "length" and answer.strip():
            answer += OUTPUT_LIMIT_NOTICE
        cleaned = strip_thinking(answer)
        if not cleaned:
            raise InvalidModelResponseError("LLM returned an empty response.")
        return cleaned

    async def chat_structured(
        self, messages: list[dict], *, schema: dict, num_predict=None, num_ctx=None,
    ) -> dict:
        _LAST_CHAT_USAGE.set(None)
        result = await self.provider.chat_json(
            messages, schema=schema, num_predict=num_predict, num_ctx=num_ctx,
        )
        _LAST_CHAT_USAGE.set(dict(result.usage))
        return parse_structured_reply(result.content)

    async def chat_stream(
        self, messages: list[dict], *, num_predict=None, num_ctx=None,
    ) -> AsyncGenerator[str, None]:
        _LAST_CHAT_USAGE.set(None)
        usage: dict = {}
        provider_stream = self.provider.chat_stream(
            messages, num_predict=num_predict, num_ctx=num_ctx,
        )

        async def raw_chunks():
            try:
                async for chunk in provider_stream:
                    if chunk.done:
                        usage.update(chunk.usage)
                    if chunk.content:
                        yield chunk.content
                    if chunk.done and chunk.usage.get("done_reason") == "length":
                        yield OUTPUT_LIMIT_NOTICE
            finally:
                await provider_stream.aclose()

        visible_stream = strip_thinking_stream(raw_chunks())
        try:
            emitted = False
            async for chunk in visible_stream:
                emitted = True
                yield chunk
            if not emitted:
                raise InvalidModelResponseError("LLM returned an empty response.")
        finally:
            try:
                await visible_stream.aclose()
            finally:
                _LAST_CHAT_USAGE.set(dict(usage))



