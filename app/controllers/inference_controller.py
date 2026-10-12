"""Thin HTTP routes for inference; use-case decisions belong to services."""
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.core.streaming import encode_sse_stream
from app.domain.schema.inference_schema import (
    ChatRequest, ChatResponse, InfoResponse, LivenessResponse,
)
from app.services.inference_service import InferenceService

router = APIRouter(prefix="/api", tags=["Inference"])


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


Service = Annotated[InferenceService, Depends(get_inference_service)]


@router.get("/info", response_model=InfoResponse)
async def info(service: Service):
    return service.info()


@router.get("/health/live", response_model=LivenessResponse)
async def live(service: Service):
    return service.live()


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, service: Service):
    return await service.chat(body)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, service: Service):
    return StreamingResponse(
        encode_sse_stream(service.stream(body)),
        media_type="text/event-stream", headers={"Cache-Control": "no-cache"},
    )


