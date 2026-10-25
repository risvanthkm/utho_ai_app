"""Structured logger and FastAPI middleware for Python AI microservice."""

import logging
import sys
import time
import uuid
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Configure root logger with structured format
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("utho_ai_app")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs incoming HTTP requests, duration in milliseconds, and status codes."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        req_id = request.headers.get("X-Request-ID") or f"req-{uuid.uuid4().hex[:12]}"
        client_ip = request.client.host if request.client else "unknown"

        logger.info(
            f"[{request.method}] {request.url.path} from {client_ip} (ID: {req_id})"
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Request-ID"] = req_id

            level = logging.INFO
            if response.status_code >= 500:
                level = logging.ERROR
            elif response.status_code >= 400:
                level = logging.WARNING

            logger.log(
                level,
                f"[{request.method}] {request.url.path} -> {response.status_code} in {duration_ms:.2f}ms (ID: {req_id})",
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"[{request.method}] {request.url.path} FAILED with {exc} in {duration_ms:.2f}ms (ID: {req_id})",
                exc_info=True,
            )
            raise
