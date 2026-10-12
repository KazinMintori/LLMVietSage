"""Compose the stateless inference API and its injected model service."""
from fastapi import FastAPI

from app.controllers.inference_controller import router
from app.controllers.inference_errors_controller import invalid_request, model_failure, unsupported_mode
from app.domain.errors import InferenceError, InvalidRequestError, UnsupportedGenerationModeError
from app.services.inference_service import InferenceService, resolve_inference_service


def create_app(service: InferenceService | None = None) -> FastAPI:
    inference_service = resolve_inference_service(service)
    application = FastAPI(
        title="VietSage LLM Inference",
        description="Stateless LLM inference: chat, streaming and structured JSON from caller-supplied context/history.",
        version="0.5.0",
        lifespan=inference_service.lifespan,
    )
    application.state.inference_service = inference_service
    application.add_exception_handler(InferenceError, model_failure)
    application.add_exception_handler(UnsupportedGenerationModeError, unsupported_mode)
    application.add_exception_handler(InvalidRequestError, invalid_request)
    application.include_router(router)
    return application


app = create_app()
