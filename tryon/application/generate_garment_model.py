"""
Caso de uso: generar el modelo 3D de una prenda.

Flujo:
1. Comprobar en el catálogo (Digital Closet) que la prenda existe, es del usuario y es un
   accesorio soportado.
2. Obtener su foto actual.
3. Registrar el modelo como PROCESSING (con la versión de la foto) y guardar la foto.
4. Generar el .glb y su miniatura y guardarlos con el ID de la prenda -> READY.
   Si falla -> FAILED con el motivo, y se propaga el error.

Una prenda no puede tener dos generaciones a la vez (GenerationInProgress).
"""
import threading
from dataclasses import dataclass
from typing import Optional

from ..domain.errors import GenerationFailed, UnsupportedAccessory, GenerationInProgress
from ..domain.garment_model import AccessoryCategory, GarmentId, GarmentModel
from ..domain.ports import (Credentials, GarmentCatalog, GenerationOptions, ModelGenerator,
                            ModelRepository, PhotoUpload)


@dataclass(frozen=True)
class GenerateGarmentModelCommand:
    garment_id: str
    credentials: Credentials
    # Si se indica, manda sobre la subcategoría del Digital Closet (útil si no la tiene).
    category: Optional[AccessoryCategory] = None
    options: GenerationOptions = GenerationOptions()
    # Solo modo local: la foto viaja en la misma petición.
    upload: Optional[PhotoUpload] = None


class GenerateGarmentModel:

    def __init__(self, catalog: GarmentCatalog, repository: ModelRepository, generator: ModelGenerator):
        self._catalog = catalog
        self._repository = repository
        self._generator = generator
        self._in_progress: set[str] = set()
        self._lock = threading.Lock()

    def execute(self, command: GenerateGarmentModelCommand) -> GarmentModel:
        garment_id = GarmentId(command.garment_id)
        garment = self._catalog.get_garment(garment_id, command.credentials)
        category = command.category or garment.category
        if category is None:
            raise UnsupportedAccessory(
                f"La subcategoría '{garment.subcategory or 'sin subcategoría'}' no se puede generar en 3D. "
                "Por ahora: lentes (GLASSES, SUNGLASSES) y gorras o sombreros (HAT, CAP). "
                "También puedes indicar category=glasses o category=hat.")

        self._reserve(garment_id)
        try:
            photo = self._catalog.get_photo(garment_id, command.credentials, command.upload)
            model = GarmentModel.start(garment_id, category, photo.image_version)
            self._repository.save(model)
            self._repository.store_photo(garment_id, photo)
            try:
                generated = self._generator.generate(photo, category, command.options)
                self._repository.store_model_file(garment_id, generated.file_path)
                if generated.thumbnail_png:
                    self._repository.store_thumbnail(garment_id, generated.thumbnail_png)
            except GenerationFailed as exc:
                model.fail(exc.message)
                self._repository.save(model)
                raise
            except Exception as exc:          # error inesperado: se registra y se propaga
                model.fail(f"Error inesperado: {exc}")
                self._repository.save(model)
                raise
            model.complete(generated.job_id, generated.details)
            self._repository.save(model)
            return model
        finally:
            self._release(garment_id)

    # Una generación por prenda a la vez (la GPU ya se serializa en el motor).
    def _reserve(self, garment_id: GarmentId) -> None:
        with self._lock:
            if garment_id.value in self._in_progress:
                raise GenerationInProgress("Ya se está generando el modelo de esta prenda. Espera a que termine.")
            self._in_progress.add(garment_id.value)

    def _release(self, garment_id: GarmentId) -> None:
        with self._lock:
            self._in_progress.discard(garment_id.value)
