"""
Caso de uso: guardar cómo se coloca en la cara el modelo 3D de una prenda (ar_fit).

La app calibra el modelo en el teléfono (subir, adelantar, escalar, inclinar) y lo envía.
Se comprueba en el catálogo que la prenda es del usuario y que su modelo está listo y se
puede probar en RA. El ajuste queda con el modelo: regenerarlo lo descarta, porque la
geometría nueva puede asentar distinto.

Si una generación empieza mientras tanto, su guardado final manda: el modelo nuevo
queda sin calibrar, que es lo correcto.
"""
from ..domain.errors import ModelNotFound, ModelNotReady
from ..domain.garment_model import ArSupportPolicy, GarmentId
from ..domain.ports import Credentials, GarmentCatalog, ModelRepository
from .get_garment_model import GarmentModelView


class CalibrateArFit:

    def __init__(self, catalog: GarmentCatalog, repository: ModelRepository,
                 ar_policy: ArSupportPolicy = ArSupportPolicy()):
        self._catalog = catalog
        self._repository = repository
        self._ar_policy = ar_policy

    def execute(self, garment_id: str, credentials: Credentials, fit: dict) -> GarmentModelView:
        gid = GarmentId(garment_id)
        garment = self._catalog.get_garment(gid, credentials)
        model = self._repository.get(gid)
        if model is None:
            raise ModelNotFound("Esta prenda todavía no tiene modelo 3D. Genéralo primero.")
        if not model.is_ready:
            raise ModelNotReady("Solo se puede calibrar un modelo listo.", status=model.status.value)
        model.calibrate(self._ar_policy.calibrated_fit(model, fit))
        self._repository.save(model)
        return GarmentModelView(
            model=model,
            is_stale=model.is_stale(garment.image_version),
            badge=model.badge(self._ar_policy),
            has_thumbnail=self._repository.has_thumbnail(gid),
            ar_fit=self._ar_policy.fit_for(model),
        )
