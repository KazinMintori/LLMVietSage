"""The narrow provider contract used by generation services and test fakes."""
from collections.abc import AsyncGenerator
from typing import Protocol

from app.domain.models.generation import ProviderChunk, ProviderReply


class ModelProvider(Protocol):
    provider_name: str

    async def startup(self) -> None: ...
    async def shutdown(self) -> None: ...

    async def chat(
        self, messages: list[dict], *, num_predict: int | None = None, num_ctx: int | None = None,
    ) -> ProviderReply: ...

    async def chat_json(
        self, messages: list[dict], *, schema: dict,
        num_predict: int | None = None, num_ctx: int | None = None,
    ) -> ProviderReply: ...

    def chat_stream(
        self, messages: list[dict], *, num_predict: int | None = None, num_ctx: int | None = None,
    ) -> AsyncGenerator[ProviderChunk, None]: ...
