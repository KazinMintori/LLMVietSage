"""Stateless inference use cases: prompt, mode selection, lifecycle and events."""
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app.domain.errors import ModelUnavailableError, UnsupportedGenerationModeError
from app.domain.models.generation import StreamEvent
from app.domain.schema.inference_schema import (
    ChatRequest, ChatResponse, InfoResponse, LivenessResponse,
)
from app.services.generation_budget_service import generation_options, requests_extended_output
from app.services.input_validation_service import validate_chat_request
from app.services.llm_service import LLMService, current_chat_usage
from config.prompts import CHAT_SYSTEM_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)


def prepare_chat(body: ChatRequest) -> tuple[list[dict], dict]:
    """Prepare caller-supplied turns without retrieval, persistence or mutation."""
    validate_chat_request(body)
    if body.messages is not None:
        turns = [turn.model_dump() for turn in body.messages]
    else:
        turns = [{"role": "user", "content": body.message}]
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}, *turns]
    long_request = requests_extended_output(turns[-1]["content"])
    options = generation_options(messages, long_request=long_request)
    return messages, options


class InferenceService:
    def __init__(self, model_service: LLMService) -> None:
        self.model_service = model_service

    @asynccontextmanager
    async def lifespan(self, _app):
        await self.model_service.startup()
        try:
            if settings.ollama_warmup_enabled:
                try:
                    await self.model_service.warmup()
                except Exception as exc:
                    logger.warning("Model warmup unavailable: %s", type(exc).__name__)
            yield
        finally:
            await self.model_service.shutdown()

    def info(self) -> InfoResponse:
        return InfoResponse(
            app="VietSage LLM", provider=self.model_service.provider_name,
            model=settings.ollama_chat_model, history="caller_supplied", rag_enabled=False,
        )

    def live(self) -> LivenessResponse:
        return LivenessResponse(status="ok")

    async def chat(self, body: ChatRequest) -> ChatResponse:
        messages, options = prepare_chat(body)
        if body.json_schema is not None:
            answer = await self.model_service.chat_structured(
                messages, schema=body.json_schema, **options,
            )
        else:
            answer = await self.model_service.chat(messages, **options)
        return ChatResponse(reply=answer, model=settings.ollama_chat_model, usage=current_chat_usage())

    def stream(self, body: ChatRequest) -> AsyncGenerator[StreamEvent, None]:
        # Check before response creation; model iteration stays in the streaming task.
        if body.json_schema is not None:
            raise UnsupportedGenerationModeError("Structured JSON is supported by /api/chat.")
        messages, options = prepare_chat(body)
        return self._stream_events(messages, options)

    async def _stream_events(self, messages: list[dict], options: dict):
        chunks = self.model_service.chat_stream(messages, **options)
        received = False
        try:
            async for chunk in chunks:
                received = True
                yield StreamEvent("token", {"text": chunk})
            if not received:
                yield StreamEvent("error", {"message": "The model returned an empty response."})
                return
            yield StreamEvent("done", {"model": settings.ollama_chat_model, "usage": current_chat_usage()})
        except ModelUnavailableError as exc:
            logger.warning("Inference stream failed: %s", type(exc).__name__)
            yield StreamEvent("error", {"message": "The model stream was interrupted. Please retry."})
        finally:
            await chunks.aclose()



def resolve_inference_service(service: InferenceService | None) -> InferenceService:
    if service is not None:
        return service
    from app.services.ollama_service import OllamaService
    return InferenceService(LLMService(OllamaService()))


def public_model_failure(exc: Exception) -> str:
    logger.warning("Inference provider failed: %s", type(exc).__name__)
    return "The model is unavailable. Please retry."
