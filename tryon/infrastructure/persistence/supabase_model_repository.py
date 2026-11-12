import json
from pathlib import Path

import httpx

from ...domain.errors import ModelStorageUnavailable
from ...domain.garment_model import GarmentId, GarmentModel, ModelVariant
from ...domain.ports import GarmentPhoto
from .file_model_repository import METADATA_FILE, MODEL_FILES, THUMBNAIL_FILE, _from_dict, _to_dict

BUCKET = "accessory-models"


class SupabaseModelRepository:
    """
    Adaptador del puerto ModelRepository sobre Supabase Storage (bucket privado).

    Mismos archivos por prenda que en disco: accessory-models/{garment_id}/model.glb,
    model_ar.glb (variante RA), thumbnail.png, metadata.json y source-photo.
    La API comprueba que la prenda es del usuario antes de leer cualquier archivo.
    """

    def __init__(self, url: str, key: str):
        self._url = f"{url}/storage/v1/object"
        self._headers = {"apikey": key}
        if key.startswith("eyJ"):
            self._headers["Authorization"] = f"Bearer {key}"

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        headers = self._headers | kwargs.pop("headers", {})
        try:
            return httpx.request(method, f"{self._url}/{path}", headers=headers, timeout=30, **kwargs)
        except httpx.RequestError as exc:
            raise ModelStorageUnavailable("No se pudo conectar al almacenamiento 3D.") from exc

    def _read(self, garment_id: GarmentId, name: str, *, info: bool = False) -> bytes | None:
        route = "info/authenticated" if info else "authenticated"
        response = self._request("GET", f"{route}/{BUCKET}/{garment_id.value}/{name}")
        if response.is_success:
            return response.content
        if response.status_code in (400, 404):
            try:
                payload = response.json()
            except ValueError:
                payload = {}
            if isinstance(payload, dict) and (
                payload.get("code") in ("NoSuchKey", "ObjectNotFound")
                or payload.get("error") in ("NoSuchKey", "Object not found")
                or (str(payload.get("statusCode")) == "404"
                    and str(payload.get("message", "")).lower() == "object not found")
            ):
                return None
        raise ModelStorageUnavailable("No se pudo leer un archivo del almacenamiento 3D.")

    def _write(self, garment_id: GarmentId, name: str, content: bytes, mime: str) -> None:
        response = self._request(
            "POST", f"{BUCKET}/{garment_id.value}/{name}", content=content,
            headers={"Content-Type": mime, "x-upsert": "true"},
        )
        if not response.is_success:
            raise ModelStorageUnavailable("No se pudo guardar un archivo en el almacenamiento 3D.")

    def get(self, garment_id: GarmentId) -> GarmentModel | None:
        data = self._read(garment_id, METADATA_FILE)
        if data is None:
            return None
        try:
            return _from_dict(json.loads(data))
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelStorageUnavailable("Los metadatos del modelo 3D no son válidos.") from exc

    def save(self, model: GarmentModel) -> None:
        data = json.dumps(_to_dict(model), ensure_ascii=False).encode()
        self._write(model.garment_id, METADATA_FILE, data, "application/json")

    def store_photo(self, garment_id: GarmentId, photo: GarmentPhoto) -> None:
        self._write(garment_id, "source-photo", photo.data, photo.content_type)

    def store_model_file(self, garment_id: GarmentId, generated_file: Path,
                         variant: ModelVariant = ModelVariant.FULL) -> None:
        self._write(garment_id, MODEL_FILES[variant], generated_file.read_bytes(), "model/gltf-binary")

    def model_file(self, garment_id: GarmentId,
                   variant: ModelVariant = ModelVariant.FULL) -> bytes | None:
        return self._read(garment_id, MODEL_FILES[variant])

    def store_thumbnail(self, garment_id: GarmentId, png: bytes) -> None:
        self._write(garment_id, THUMBNAIL_FILE, png, "image/png")

    def thumbnail_file(self, garment_id: GarmentId) -> bytes | None:
        return self._read(garment_id, THUMBNAIL_FILE)

    def has_thumbnail(self, garment_id: GarmentId) -> bool:
        return self._read(garment_id, THUMBNAIL_FILE, info=True) is not None
