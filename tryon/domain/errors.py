"""
Errores del dominio.

Cada error tiene un `code` estable (lo que lee la app) y un mensaje en español para el
usuario. La capa HTTP traduce cada tipo a un código de estado; el dominio no sabe de HTTP.
"""


class DomainError(Exception):
    """Base de todos los errores de negocio del servicio."""

    code = "DOMAIN_ERROR"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


# --- La petición no es válida ------------------------------------------------------------

class InvalidGarmentId(DomainError):
    code = "INVALID_GARMENT_ID"


class InvalidPhoto(DomainError):
    code = "INVALID_PHOTO"


class UnsupportedAccessory(DomainError):
    """La prenda no es un accesorio soportado (por ahora: lentes y gorras o sombreros)."""

    code = "UNSUPPORTED_ACCESSORY"


# --- La prenda no está disponible --------------------------------------------------------

class NotAuthenticated(DomainError):
    code = "UNAUTHENTICATED"


class GarmentNotFound(DomainError):
    """No existe, está borrada o pertenece a otro usuario (no se distingue a propósito)."""

    code = "GARMENT_NOT_FOUND"


class GarmentWithoutPhoto(DomainError):
    code = "GARMENT_WITHOUT_PHOTO"


# --- Estado del modelo 3D ----------------------------------------------------------------

class ModelNotFound(DomainError):
    code = "MODEL_NOT_FOUND"


class ModelNotReady(DomainError):
    """Existe, pero se está generando o la última generación falló."""

    code = "MODEL_NOT_READY"

    def __init__(self, message: str, status: str):
        super().__init__(message)
        self.status = status


class GenerationInProgress(DomainError):
    code = "GENERATION_IN_PROGRESS"


# --- Dependencias externas ---------------------------------------------------------------

class CatalogUnavailable(DomainError):
    """El Digital Closet no responde o respondió algo inesperado."""

    code = "CATALOG_UNAVAILABLE"


class GenerationFailed(DomainError):
    code = "GENERATION_FAILED"


class ModelStorageUnavailable(DomainError):
    code = "MODEL_STORAGE_UNAVAILABLE"


class InsufficientGpuMemory(GenerationFailed):
    code = "INSUFFICIENT_GPU_MEMORY"
