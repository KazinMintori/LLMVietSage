"""Branch-free HTTP formatting for already classified application failures."""
from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain.errors import InferenceError, InvalidRequestError, UnsupportedGenerationModeError
from app.services.inference_service import public_model_failure


async def model_failure(_request: Request, exc: InferenceError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": public_model_failure(exc)})


async def unsupported_mode(_request: Request, exc: UnsupportedGenerationModeError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def invalid_request(_request: Request, exc: InvalidRequestError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.issues})
