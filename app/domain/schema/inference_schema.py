"""Request validation and public response contracts for model inference."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str | None = Field(default=None, min_length=1)
    messages: list[ChatMessage] | None = Field(default=None, min_length=1)
    json_schema: dict | None = None


class ChatResponse(BaseModel):
    reply: str | dict[str, Any]
    model: str
    usage: dict[str, Any]




class InfoResponse(BaseModel):
    app: str
    provider: str
    model: str
    history: str
    rag_enabled: bool


class LivenessResponse(BaseModel):
    status: str
