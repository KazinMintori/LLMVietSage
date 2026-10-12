"""Inference failures independent of HTTP status codes and provider libraries."""


class InferenceError(RuntimeError):
    """Base class for failures understood by the inference application."""


class ModelUnavailableError(InferenceError):
    """The inference provider cannot complete the request."""


class InvalidModelResponseError(ModelUnavailableError):
    """The provider returned an unusable or incomplete response."""


class UnsupportedGenerationModeError(InferenceError):
    """The requested output mode is unavailable on this endpoint."""


class InvalidRequestError(InferenceError):
    """Semantic input violations; declarative schema errors are separate."""

    def __init__(self, issues: list[dict]) -> None:
        super().__init__("Invalid inference request.")
        self.issues = issues
