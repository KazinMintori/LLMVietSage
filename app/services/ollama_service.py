"""Ollama HTTP protocol, connection lifecycle and provider error translation."""
import json
import math
from collections.abc import AsyncGenerator

import httpx

from app.domain.errors import InvalidModelResponseError, ModelUnavailableError
from app.domain.models.generation import ProviderChunk, ProviderReply
from config.settings import settings

_USAGE_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_cached_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
)


def _usage(payload: dict) -> dict:
    usage = {}
    for field in _USAGE_FIELDS:
        value = payload.get(field)
        if (type(value) is int or type(value) is float and math.isfinite(value)) and value >= 0:
            usage[field] = value
    reason = payload.get("done_reason")
    if isinstance(reason, str) and reason:
        usage["done_reason"] = reason
    return usage


def _content(payload: dict) -> str:
    message = payload.get("message", {})
    if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
        raise InvalidModelResponseError("Ollama returned invalid message content.")
    return message.get("content", "")


class OllamaService:
    """Return provider data; output policy belongs to the application service."""

    provider_name = "ollama"

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.chat_model = settings.ollama_chat_model
        self.keep_alive = settings.ollama_keep_alive
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _chat_options(*, num_predict: int | None = None, num_ctx: int | None = None) -> dict:
        return {
            "num_ctx": settings.ollama_num_ctx if num_ctx is None else num_ctx,
            "temperature": settings.ollama_temperature,
            "num_predict": settings.ollama_num_predict if num_predict is None else num_predict,
        }

    async def startup(self) -> None:
        if self._client is None or self._client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=min(10, settings.api_max_connections),
                max_connections=settings.api_max_connections,
            )
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    settings.api_timeout_seconds,
                    connect=settings.api_connect_timeout_seconds,
                ),
                limits=limits,
                transport=httpx.AsyncHTTPTransport(retries=1, limits=limits),
            )

    async def shutdown(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def _get_client(self) -> httpx.AsyncClient:
        await self.startup()
        assert self._client is not None
        return self._client

    async def _post(self, path: str, payload: dict) -> dict:
        try:
            client = await self._get_client()
            response = await client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError("Ollama request failed.") from exc
        except (TypeError, ValueError) as exc:
            raise InvalidModelResponseError("Ollama returned invalid JSON.") from exc
        if not isinstance(result, dict) or result.get("error"):
            raise InvalidModelResponseError("Ollama returned an invalid response.")
        return result

    def _payload(self, messages: list[dict], *, stream: bool, num_predict=None, num_ctx=None) -> dict:
        return {
            "model": self.chat_model,
            "messages": messages,
            "stream": stream,
            "think": settings.ollama_think,
            "keep_alive": self.keep_alive,
            "options": self._chat_options(num_predict=num_predict, num_ctx=num_ctx),
        }

    async def chat(self, messages: list[dict], *, num_predict=None, num_ctx=None) -> ProviderReply:
        result = await self._post("/api/chat", self._payload(
            messages, stream=False, num_predict=num_predict, num_ctx=num_ctx,
        ))
        return ProviderReply(content=_content(result), usage=_usage(result))

    async def chat_json(
        self, messages: list[dict], *, schema: dict, num_predict=None, num_ctx=None,
    ) -> ProviderReply:
        payload = self._payload(messages, stream=False, num_predict=num_predict, num_ctx=num_ctx)
        payload["format"] = schema or "json"
        result = await self._post("/api/chat", payload)
        return ProviderReply(content=_content(result), usage=_usage(result))

    async def chat_stream(
        self, messages: list[dict], *, num_predict=None, num_ctx=None,
    ) -> AsyncGenerator[ProviderChunk, None]:
        try:
            client = await self._get_client()
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=self._payload(messages, stream=True, num_predict=num_predict, num_ctx=num_ctx),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if not isinstance(chunk, dict) or chunk.get("error"):
                        raise InvalidModelResponseError("Ollama streaming request failed.")
                    done = chunk.get("done", False)
                    if not isinstance(done, bool):
                        raise InvalidModelResponseError("Ollama returned an invalid completion flag.")
                    yield ProviderChunk(
                        content=_content(chunk), done=done, usage=_usage(chunk) if done else {},
                    )
                    if done:
                        return
        except httpx.HTTPError as exc:
            raise ModelUnavailableError("Ollama stream failed.") from exc
        except (TypeError, ValueError) as exc:
            raise InvalidModelResponseError("Ollama returned invalid stream JSON.") from exc
        raise InvalidModelResponseError("Ollama stream ended before completion.")



