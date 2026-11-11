"""
MODO LOCAL (temporal): catálogo sin Digital Closet.

Sirve para probar mientras el Digital Closet no está desplegado. Se activa solo cuando
DC_API_URL está vacía.
- No verifica que la prenda exista ni a quién pertenece: confía en el garment_id recibido.
- La foto viaja en la misma petición (campo `image`) y la categoría se indica a mano.

Para QUITAR el modo local:
1. Borrar este archivo.
2. En tryon/bootstrap.py, borrar la rama `else` de build_catalog().
3. (Opcional) En el router, quitar el campo `image` de POST /garments/{id}/model.
Nada del dominio ni de los casos de uso cambia.
"""
from typing import Optional

from ...domain.errors import GarmentWithoutPhoto, InvalidPhoto
from ...domain.garment_model import GarmentId
from ...domain.ports import Credentials, GarmentPhoto, GarmentSnapshot, PhotoUpload

ALLOWED_TYPES = ("image/jpeg", "image/png", "image/webp")


class LocalGarmentCatalog:

    def __init__(self, max_photo_mb: int = 10):
        self._max_photo_bytes = max_photo_mb * 1024 * 1024

    def get_garment(self, garment_id: GarmentId, credentials: Credentials) -> GarmentSnapshot:
        # Sin catálogo no se conoce la subcategoría ni la versión de la foto.
        return GarmentSnapshot(garment_id=garment_id, subcategory=None, category=None, image_version=None)

    def get_garments(self, garment_ids: list[GarmentId],
                     credentials: Credentials) -> dict[GarmentId, Optional[GarmentSnapshot]]:
        return {gid: self.get_garment(gid, credentials) for gid in garment_ids}

    def get_photo(self, garment_id: GarmentId, credentials: Credentials,
                  upload: Optional[PhotoUpload]) -> GarmentPhoto:
        if upload is None:
            raise GarmentWithoutPhoto("Modo local: envía la foto en el campo 'image'.")
        if upload.content_type not in ALLOWED_TYPES:
            raise InvalidPhoto("Formato no soportado. Usa JPG, PNG o WebP.")
        if len(upload.data) > self._max_photo_bytes:
            raise InvalidPhoto("La foto supera el tamaño máximo permitido.")
        return GarmentPhoto(data=upload.data, image_version=None, content_type=upload.content_type)
