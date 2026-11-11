"""
Casos de uso de consulta del modelo 3D de una prenda.

Antes de responder se pregunta al catálogo por la prenda: así solo su dueño puede ver o
descargar el modelo, y se sabe si la foto cambió desde que se generó (is_stale).
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..domain.errors import InvalidGarmentId, ModelNotFound, ModelNotReady
from ..domain.garment_model import ArSupportPolicy, GarmentId, GarmentModel, ModelBadge
from ..domain.ports import Credentials, GarmentCatalog, ModelRepository

MAX_BATCH = 100


@dataclass(frozen=True)
class GarmentModelView:
    model: GarmentModel
    is_stale: bool
    badge: Optional[ModelBadge]
    file_path: Optional[Path] = None        # .glb, solo si está READY
    thumbnail_path: Optional[Path] = None   # miniatura, si existe


@dataclass(frozen=True)
class GarmentModelSummary:
    """Una fila de la consulta en lote (pantalla del armario)."""

    garment_id: str
    found: bool                              # False = no existe o no es del usuario
    model: Optional[GarmentModel]
    is_stale: bool
    badge: Optional[ModelBadge]
    has_thumbnail: bool


class GetGarmentModel:

    def __init__(self, catalog: GarmentCatalog, repository: ModelRepository,
                 ar_policy: ArSupportPolicy = ArSupportPolicy()):
        self._catalog = catalog
        self._repository = repository
        self._ar_policy = ar_policy

    def info(self, garment_id: str, credentials: Credentials) -> GarmentModelView:
        """Estado del modelo (cualquier estado)."""
        gid = GarmentId(garment_id)
        garment = self._catalog.get_garment(gid, credentials)
        model = self._repository.get(gid)
        if model is None:
            raise ModelNotFound("Esta prenda todavía no tiene modelo 3D. Genéralo primero.")
        return GarmentModelView(
            model=model,
            is_stale=model.is_stale(garment.image_version),
            badge=model.badge(self._ar_policy),
            file_path=self._repository.model_file(gid) if model.is_ready else None,
            thumbnail_path=self._repository.thumbnail_file(gid),
        )

    def file(self, garment_id: str, credentials: Credentials) -> GarmentModelView:
        """El .glb listo para descargar."""
        view = self.info(garment_id, credentials)
        if not view.model.is_ready or view.file_path is None:
            raise ModelNotReady(
                "El modelo 3D aún no está listo." if view.model.status.value == "PROCESSING"
                else f"La última generación falló: {view.model.error}",
                status=view.model.status.value)
        return view

    def thumbnail(self, garment_id: str, credentials: Credentials) -> GarmentModelView:
        """La miniatura (foto sin fondo) para la tarjeta de la prenda."""
        view = self.info(garment_id, credentials)
        if view.thumbnail_path is None:
            raise ModelNotFound("Esta prenda no tiene miniatura. Se crea al generar el modelo 3D.")
        return view

    def many(self, garment_ids: list[str], credentials: Credentials) -> list[GarmentModelSummary]:
        """
        Estado y etiqueta de varias prendas en una sola llamada, en el orden pedido.
        Las prendas que no existen o son de otro usuario vuelven con found=False.
        """
        if not 1 <= len(garment_ids) <= MAX_BATCH:
            raise InvalidGarmentId(f"Envía entre 1 y {MAX_BATCH} IDs de prenda.")
        ids = [GarmentId(i) for i in garment_ids]
        if len(set(ids)) != len(ids):
            raise InvalidGarmentId("Hay IDs de prenda repetidos.")
        garments = self._catalog.get_garments(ids, credentials)
        result = []
        for gid in ids:
            garment = garments.get(gid)
            model = self._repository.get(gid) if garment is not None else None
            result.append(GarmentModelSummary(
                garment_id=gid.value,
                found=garment is not None,
                model=model,
                is_stale=bool(model and model.is_stale(garment.image_version)),
                badge=model.badge(self._ar_policy) if model else None,
                has_thumbnail=garment is not None and self._repository.thumbnail_file(gid) is not None,
            ))
        return result
