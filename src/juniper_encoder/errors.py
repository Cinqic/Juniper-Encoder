"""Public, machine-readable errors and deterministic precedence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import ERROR_PRECEDENCE


@dataclass
class EncoderError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"error": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def invalid_input(message: str, **details: Any) -> EncoderError:
    return EncoderError("INVALID_INPUT", message, details)


def input_too_long(message: str, **details: Any) -> EncoderError:
    return EncoderError("INPUT_TOO_LONG", message, details)


def invalid_registry(message: str, **details: Any) -> EncoderError:
    return EncoderError("INVALID_REGISTRY", message, details)


def index_mismatch(message: str, **details: Any) -> EncoderError:
    return EncoderError("INDEX_MISMATCH", message, details)


def numerical_error(message: str, **details: Any) -> EncoderError:
    return EncoderError("NUMERICAL_ERROR", message, details)


def precedence_document() -> list[str]:
    return list(ERROR_PRECEDENCE)
