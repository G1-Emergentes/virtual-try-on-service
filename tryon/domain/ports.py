"""
Puertos: lo que el dominio necesita del exterior, sin decir cómo se hace.

La infraestructura los implementa (adaptadores). Gracias a esto:
- los casos de uso se prueban con dobles, sin GPU ni red;
- cambiar el Digital Closet por otra fuente, o el disco por un bucket, no toca el dominio.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from .garment_model import AccessoryCategory, GarmentId, GarmentModel, ModelVariant


# --- Objetos que cruzan los puertos ------------------------------------------------------

@dataclass(frozen=True)
class Credentials:
    """Token del usuario (Authorization: Bearer ...). Se reenvía al Digital Closet."""

    bearer_token: Optional[str] = None


@dataclass(frozen=True)
class PhotoUpload:
    """Foto enviada en la misma petición. Solo la usa el modo local."""

    data: bytes
    content_type: str


@dataclass(frozen=True)
class GarmentSnapshot:
    """Lo que este servicio necesita saber de una prenda del Digital Closet."""

    garment_id: GarmentId
    subcategory: Optional[str]
    category: Optional[AccessoryCategory]   # None si la subcategoría no se reconoce
    image_version: Optional[int]            # None si aún no tiene foto (o se desconoce)


@dataclass(frozen=True)
class GarmentPhoto:
    data: bytes
    image_version: Optional[int]
    content_type: str = "image/jpeg"


@dataclass(frozen=True)
class GenerationOptions:
    """Ajustes opcionales del material. None = que el motor decida."""

    lens_type: Optional[str] = None
    frame_material: Optional[str] = None


@dataclass(frozen=True)
class GeneratedModel:
    file_path: Path          # .glb temporal; el repositorio lo guarda en su lugar definitivo
    job_id: str
    details: dict = field(default_factory=dict)
    thumbnail_png: Optional[bytes] = None   # miniatura: la foto sin fondo, PNG transparente
    # Variante para RA (sin patillas), solo en lentes. ar_details describe cómo se creó
    # (p. ej. {"method": "hinge"} o {"method": "crop_temples"} si se usó el respaldo).
    ar_file_path: Optional[Path] = None
    ar_details: dict = field(default_factory=dict)


# --- Puertos -----------------------------------------------------------------------------

class GarmentCatalog(Protocol):
    """Fuente de verdad de las prendas (el Digital Closet)."""

    def get_garment(self, garment_id: GarmentId, credentials: Credentials) -> GarmentSnapshot:
        """
        Devuelve la prenda si existe y pertenece al usuario.
        Errores: NotAuthenticated, GarmentNotFound, UnsupportedAccessory, CatalogUnavailable.
        """
        ...

    def get_garments(self, garment_ids: list[GarmentId],
                     credentials: Credentials) -> dict[GarmentId, Optional[GarmentSnapshot]]:
        """
        Varias prendas en una llamada (para la pantalla del armario). Cada ID devuelve su
        snapshot, o None si no existe o no es del usuario.
        Errores: NotAuthenticated, CatalogUnavailable.
        """
        ...

    def get_photo(self, garment_id: GarmentId, credentials: Credentials,
                  upload: Optional[PhotoUpload]) -> GarmentPhoto:
        """
        Devuelve la foto actual de la prenda.
        `upload` solo lo usa el modo local; el Digital Closet toma la foto de su Storage.
        Errores: GarmentWithoutPhoto, InvalidPhoto, CatalogUnavailable.
        """
        ...


class ModelRepository(Protocol):
    """Dónde se guardan los modelos 3D y su estado, indexados por el ID de la prenda."""

    def get(self, garment_id: GarmentId) -> Optional[GarmentModel]:
        ...

    def save(self, model: GarmentModel) -> None:
        ...

    def store_photo(self, garment_id: GarmentId, photo: GarmentPhoto) -> None:
        """Guarda la foto usada, para trazabilidad (de qué imagen salió el modelo)."""
        ...

    def store_model_file(self, garment_id: GarmentId, generated_file: Path,
                         variant: ModelVariant = ModelVariant.FULL) -> None:
        """Guarda el .glb de esa variante (reemplaza el anterior)."""
        ...

    def model_file(self, garment_id: GarmentId,
                   variant: ModelVariant = ModelVariant.FULL) -> Path | bytes | None:
        """El .glb de esa variante, o None si no existe."""
        ...

    def store_thumbnail(self, garment_id: GarmentId, png: bytes) -> None:
        ...

    def thumbnail_file(self, garment_id: GarmentId) -> Path | bytes | None:
        ...

    def has_thumbnail(self, garment_id: GarmentId) -> bool:
        ...


class ModelGenerator(Protocol):
    """Convierte una foto en un modelo 3D (.glb). Implementación: SPAR3D + posprocesado."""

    def generate(self, photo: GarmentPhoto, category: AccessoryCategory,
                 options: GenerationOptions) -> GeneratedModel:
        """Errores: InvalidPhoto, InsufficientGpuMemory, GenerationFailed."""
        ...
