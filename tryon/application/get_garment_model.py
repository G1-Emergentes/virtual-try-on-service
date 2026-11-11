"""
Casos de uso de consulta del modelo 3D de una prenda.

Antes de responder se pregunta al catálogo por la prenda: así solo su dueño puede ver o
descargar el modelo, y se sabe si la foto cambió desde que se generó (is_stale).
"""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from ..domain.errors import InvalidGarmentId, ModelNotFound, ModelNotReady
from ..domain.garment_model import ArFit, ArSupportPolicy, GarmentId, GarmentModel, ModelBadge
from ..domain.ports import Credentials, GarmentCatalog, ModelRepository

MAX_BATCH = 100


@dataclass(frozen=True)
class GarmentModelView:
    model: GarmentModel
    is_stale: bool
    badge: Optional[ModelBadge]
    file_path: Path | bytes | None = None
    thumbnail_path: Path | bytes | None = None
    has_thumbnail: bool = False
    ar_fit: Optional[ArFit] = None           # cómo se coloca en la cara (solo AR_READY)


@dataclass(frozen=True)
class GarmentModelSummary:
    """Una fila de la consulta en lote (pantalla del armario)."""

    garment_id: str
    found: bool                              # False = no existe o no es del usuario
    model: Optional[GarmentModel]
    is_stale: bool
    badge: Optional[ModelBadge]
    has_thumbnail: bool
    ar_fit: Optional[ArFit] = None


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
            has_thumbnail=self._repository.has_thumbnail(gid),
            ar_fit=self._ar_policy.fit_for(model),
        )

    def file(self, garment_id: str, credentials: Credentials) -> GarmentModelView:
        """El .glb listo para descargar."""
        view = self.info(garment_id, credentials)
        asset = self._repository.model_file(view.model.garment_id) if view.model.is_ready else None
        if asset is None:
            raise ModelNotReady(
                "El modelo 3D aún no está listo." if view.model.status.value == "PROCESSING"
                else f"La última generación falló: {view.model.error}",
                status=view.model.status.value)
        return replace(view, file_path=asset)

    def thumbnail(self, garment_id: str, credentials: Credentials) -> GarmentModelView:
        """La miniatura (foto sin fondo) para la tarjeta de la prenda."""
        view = self.info(garment_id, credentials)
        asset = self._repository.thumbnail_file(view.model.garment_id) if view.has_thumbnail else None
        if asset is None:
            raise ModelNotFound("Esta prenda no tiene miniatura. Se crea al generar el modelo 3D.")
        return replace(view, thumbnail_path=asset)

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
                has_thumbnail=garment is not None and self._repository.has_thumbnail(gid),
                ar_fit=self._ar_policy.fit_for(model) if model else None,
            ))
        return result
