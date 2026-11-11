"""
Traducción de errores del dominio a respuestas HTTP.

Formato de error (el mismo que usa el Digital Closet, para que la app maneje uno solo):
    {"error": {"code": "GARMENT_NOT_FOUND", "message": "..."}}
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ...domain import errors as e

STATUS_BY_ERROR = [
    (e.InvalidGarmentId, 400),
    (e.InvalidPhoto, 400),
    (e.InvalidArFit, 400),
    (e.NotAuthenticated, 401),
    (e.GarmentNotFound, 404),
    (e.ModelNotFound, 404),
    (e.GenerationInProgress, 409),
    (e.ModelNotReady, 409),
    (e.GarmentWithoutPhoto, 422),
    (e.UnsupportedAccessory, 422),
    (e.CatalogUnavailable, 503),
    (e.ModelStorageUnavailable, 503),
    (e.InsufficientGpuMemory, 507),
    (e.GenerationFailed, 500),
]


def status_for(error: e.DomainError) -> int:
    for error_type, status in STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return status
    return 500


def error_body(error: e.DomainError) -> dict:
    body = {"code": error.code, "message": error.message}
    if isinstance(error, e.ModelNotReady):
        body["status"] = error.status
    return {"error": body}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(e.DomainError)
    async def _domain_error(_: Request, exc: e.DomainError):
        return JSONResponse(status_code=status_for(exc), content=error_body(exc))
