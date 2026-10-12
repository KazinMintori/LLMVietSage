"""Inference token/context budget policy; no HTTP provider dependency."""
import math

from config.generation import _LONG_RESPONSE_MARKERS, _MULTI_DAY_RESPONSE_PATTERN
from config.settings import settings


def estimate_tokens(text: str) -> int:
    """Estimate provider tokens without treating Vietnamese UTF-8 bytes as tokens.

    Ollama's tokenizer is model-specific, so this is only a planning heuristic.
    Three Unicode characters per token is intentionally conservative for the
    Vietnamese prompts used here while avoiding the previous 3x UTF-8
    overestimate that disabled extended answers almost all the time.
    """
    return math.ceil(len(text) / 3) if text else 0


def estimate_message_tokens(messages: list[dict]) -> int:
    # Reserve a small fixed envelope for message roles/separators. Ollama's
    # model tokenizer is not available at planning time, so this remains an
    # explicitly labelled planning heuristic rather than a token count.
    return sum(estimate_tokens(str(message.get("content") or "")) + 4 for message in messages)


def _output_budget_for_estimate(estimated_input: int, *, long_request: bool) -> int | None:
    if not long_request:
        return None
    default = settings.ollama_num_predict
    # Ollama accepts negative values (for example -1) for an unbounded output
    # mode. Preserve that configured behavior instead of replacing it with a
    # positive cap.
    if default < 0:
        return None
    desired = max(default, settings.ollama_long_num_predict)
    available = settings.ollama_num_ctx - estimated_input - settings.ollama_context_reserve_tokens
    if available < default:
        return None
    return min(desired, available)


def _context_budget_for_estimate(estimated_input: int, *, long_request: bool) -> int | None:
    # Negative provider budgets are special modes, not a negative token count.
    # Their output headroom is unknown, so preserve the configured context.
    if long_request or settings.ollama_num_predict < 0:
        return None
    configured = settings.ollama_num_ctx
    short_context = min(configured, settings.ollama_short_num_ctx)
    required = estimated_input + settings.ollama_num_predict + settings.ollama_context_reserve_tokens
    if required <= short_context:
        return short_context
    return None


def generation_options(messages: list[dict], *, long_request: bool) -> dict[str, int | None]:
    """Plan output/context together with one input-token estimate.

    ``None`` delegates to the existing configured provider default. Negative
    output budgets retain both defaults because their required output room
    cannot be estimated from a positive token cap.
    """
    if settings.ollama_num_predict < 0:
        return {"num_predict": None, "num_ctx": None}
    estimated_input = estimate_message_tokens(messages)
    return {
        "num_predict": _output_budget_for_estimate(estimated_input, long_request=long_request),
        "num_ctx": _context_budget_for_estimate(estimated_input, long_request=long_request),
    }


def output_budget_for(messages: list[dict], *, long_request: bool) -> int | None:
    """Choose extended output only when estimated input leaves enough room."""
    if not long_request or settings.ollama_num_predict < 0:
        return None
    return _output_budget_for_estimate(estimate_message_tokens(messages), long_request=long_request)


def context_budget_for(messages: list[dict], *, long_request: bool) -> int | None:
    """Reduce the KV cache only when bounded output still fits the short context."""
    if long_request or settings.ollama_num_predict < 0:
        return None
    return _context_budget_for_estimate(estimate_message_tokens(messages), long_request=long_request)


def requests_extended_output(message: str) -> bool:
    """Recognize explicit requests whose useful answer normally needs more space."""
    normalized = message.casefold()
    return (
        any(marker in normalized for marker in _LONG_RESPONSE_MARKERS)
        or bool(_MULTI_DAY_RESPONSE_PATTERN.search(normalized))
    )
