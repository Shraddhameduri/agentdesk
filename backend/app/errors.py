"""Standardized error envelope + exception handlers.

Every error response from this API has the shape:
    {"error": {"code": "<MACHINE_CODE>", "message": "<human readable>", "request_id": "<id>"}}
HTTP status codes are mapped per code; validation and auth failures use the
same envelope. No stack traces are ever returned to clients.
"""
from __future__ import annotations

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .logging_setup import get_request_id, log_event, setup_logging

logger = setup_logging()


class APIError(Exception):
    """Raised for controlled API errors; converted to the envelope by a handler."""

    def __init__(self, code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


# Well-known error codes and their HTTP status
NOT_FOUND = ("NOT_FOUND", status.HTTP_404_NOT_FOUND)
UNAUTHORIZED = ("UNAUTHORIZED", status.HTTP_401_UNAUTHORIZED)
VALIDATION_ERROR = ("VALIDATION_ERROR", status.HTTP_422_UNPROCESSABLE_ENTITY)
APPROVAL_NOT_FOUND = ("APPROVAL_NOT_FOUND", status.HTTP_404_NOT_FOUND)
APPROVAL_ALREADY_RESOLVED = ("APPROVAL_ALREADY_RESOLVED", status.HTTP_409_CONFLICT)
ORDER_NOT_FOUND = ("ORDER_NOT_FOUND", status.HTTP_404_NOT_FOUND)
TICKET_NOT_FOUND = ("TICKET_NOT_FOUND", status.HTTP_404_NOT_FOUND)
PROVIDER_UNAVAILABLE = ("PROVIDER_UNAVAILABLE", status.HTTP_503_SERVICE_UNAVAILABLE)
RATE_LIMITED = ("RATE_LIMITED", status.HTTP_429_TOO_MANY_REQUESTS)
INTERNAL_ERROR = ("INTERNAL_ERROR", status.HTTP_500_INTERNAL_SERVER_ERROR)


def error_envelope(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    rid = get_request_id()
    log_event(logger, "api_error", code=exc.code, message=exc.message, path=request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_envelope(exc.code, exc.message, rid),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    rid = get_request_id()
    details = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    log_event(logger, "validation_error", details=details, path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_envelope("VALIDATION_ERROR", details or "Invalid request payload.", rid),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = get_request_id()
    # Log the exception server-side (with traceback); the client only gets the envelope.
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_envelope(
            "INTERNAL_ERROR",
            "An unexpected error occurred. The request ID has been logged for investigation.",
            rid,
        ),
    )


def register_error_handlers(app) -> None:
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
