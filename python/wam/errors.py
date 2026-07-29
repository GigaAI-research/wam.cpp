from __future__ import annotations

from enum import IntEnum


class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    UNSUPPORTED = 3
    INCOMPATIBLE_ARTIFACT = 4
    RESOURCE_EXHAUSTED = 5
    FAILED_PRECONDITION = 6
    INFERENCE_FAILED = 7
    INTERNAL = 8


class WamError(RuntimeError):
    def __init__(self, code: int, message: str, details=None):
        super().__init__(message)
        try:
            self.code = ErrorCode(code)
        except ValueError:
            self.code = code
        self.details = details or []
