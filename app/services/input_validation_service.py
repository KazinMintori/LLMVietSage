"""Semantic request validation; schemas keep declarative type/field rules."""
from app.domain.errors import InvalidRequestError


def _reject(message: str, data: dict) -> None:
    raise InvalidRequestError([{
        "type": "value_error", "loc": ["body"], "msg": f"Value error, {message}",
        "input": data, "ctx": {"error": {}},
    }])


def validate_chat_request(body) -> None:
    if (body.message is None) == (body.messages is None):
        _reject("Send either message or messages.", body.model_dump(exclude_unset=True))
    if body.messages is not None and body.messages[-1].role != "user":
        _reject("The last message must be from the user.", body.model_dump(exclude_unset=True))
    if body.message is not None and not body.message.strip():
        _reject("Message must not be blank.", body.model_dump(exclude_unset=True))
    if body.messages is not None and any(not turn.content.strip() for turn in body.messages):
        _reject("Message contents must not be blank.", body.model_dump(exclude_unset=True))


