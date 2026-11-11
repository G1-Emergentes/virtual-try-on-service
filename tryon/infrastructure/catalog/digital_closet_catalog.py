"""
Adaptador del puerto GarmentCatalog sobre el Digital Closet Service (HTTP).

Usa endpoints que el Digital Closet YA tiene (no hay que modificarlo):
- GET {DC_API_URL}/garments/{id}               -> la prenda (GarmentReadModel)
- GET {DC_API_URL}/garments/{id}/image-access  -> URL firmada de su foto (5 minutos)
- POST {DC_API_URL}/garments/resolve           -> varias prendas a la vez (hasta 100)

Autenticación: se reenvía el token del usuario (Authorization: Bearer ...), igual que su
visualization-worker. El Digital Closet lo valida contra IAM y responde 404 si la prenda
no es de ese usuario, así que la pertenencia queda comprobada sin acceder a su base.
"""
from typing import Optional

import httpx

from ...domain.errors import (CatalogUnavailable, GarmentNotFound, GarmentWithoutPhoto,
                              InvalidPhoto, NotAuthenticated, UnsupportedAccessory)
from ...domain.garment_model import AccessoryCategory, GarmentId
from ...domain.ports import Credentials, GarmentPhoto, GarmentSnapshot, PhotoUpload

ACCESSORY_CATEGORY = "ACCESSORY"


class DigitalClosetCatalog:

    def __init__(self, base_url: str, timeout_s: float = 15.0, max_photo_mb: int = 10,
                 transport: Optional[httpx.BaseTransport] = None):
        # base_url, p. ej. http://host.docker.internal:8081/api/v1
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._max_photo_bytes = max_photo_mb * 1024 * 1024
        self._transport = transport          # solo para pruebas (httpx.MockTransport)

    # --- Puerto GarmentCatalog -------------------------------------------------------------

    def get_garment(self, garment_id: GarmentId, credentials: Credentials) -> GarmentSnapshot:
        body = self._request_json("GET", f"/garments/{garment_id}", credentials)
        if body.get("category") != ACCESSORY_CATEGORY:
            raise UnsupportedAccessory(
                f"La prenda es de la categoría {body.get('category')}; solo se generan accesorios.")
        return _snapshot(garment_id, body)

    def get_garments(self, garment_ids: list[GarmentId],
                     credentials: Credentials) -> dict[GarmentId, Optional[GarmentSnapshot]]:
        body = self._request_json("POST", "/garments/resolve", credentials,
                                  payload={"ids": [g.value for g in garment_ids]})
        by_id = {g.value: g for g in garment_ids}
        result: dict[GarmentId, Optional[GarmentSnapshot]] = {g: None for g in garment_ids}
        for row in body.get("results") or []:
            gid = by_id.get(str(row.get("id", "")).lower())
            if gid is None or row.get("status") == "NOT_FOUND":
                continue
            garment = row.get("garment")
            # UNAVAILABLE: es del usuario, pero el Digital Closet no manda sus datos
            result[gid] = _snapshot(gid, garment) if garment else GarmentSnapshot(gid, None, None, None)
        return result

    def get_photo(self, garment_id: GarmentId, credentials: Credentials,
                  upload: Optional[PhotoUpload]) -> GarmentPhoto:
        if upload is not None:
            # La foto oficial es la del armario; aceptar otra rompería la trazabilidad.
            raise InvalidPhoto("La foto se toma del Digital Closet. No envíes una imagen en esta petición.")
        access = self._request_json("GET", f"/garments/{garment_id}/image-access", credentials,
                                    not_found=GarmentWithoutPhoto("La prenda todavía no tiene foto."))
        url = access.get("url")
        if not url:
            raise GarmentWithoutPhoto("La prenda todavía no tiene foto.")
        if not url.startswith(("http://", "https://")):
            raise CatalogUnavailable("El Digital Closet devolvió una URL de foto no absoluta.")
        return GarmentPhoto(
            data=self._download(url),
            image_version=access.get("image_version"),
            content_type=access.get("mime_type") or "image/jpeg",
        )

    # --- HTTP ------------------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self._timeout, transport=self._transport)

    def _request_json(self, method: str, path: str, credentials: Credentials,
                      payload: Optional[dict] = None, not_found: Optional[Exception] = None) -> dict:
        if not credentials.bearer_token:
            raise NotAuthenticated("Falta el token del usuario (Authorization: Bearer ...).")
        try:
            with self._client() as client:
                response = client.request(method, self._base_url + path, json=payload,
                                          headers={"Authorization": f"Bearer {credentials.bearer_token}"})
        except httpx.HTTPError as exc:
            raise CatalogUnavailable(f"No se pudo contactar al Digital Closet: {exc}")

        if response.status_code == 401:
            raise NotAuthenticated("El Digital Closet rechazó el token del usuario.")
        if response.status_code == 404:
            raise not_found or GarmentNotFound("La prenda no existe o no pertenece al usuario.")
        if response.status_code != 200:
            raise CatalogUnavailable(f"El Digital Closet respondió {response.status_code}: {_error_text(response)}")
        try:
            return response.json()
        except ValueError:
            raise CatalogUnavailable("El Digital Closet respondió algo que no es JSON.")

    def _download(self, url: str) -> bytes:
        try:
            with self._client() as client:
                response = client.get(url)
        except httpx.HTTPError as exc:
            raise CatalogUnavailable(f"No se pudo descargar la foto: {exc}")
        if response.status_code != 200:
            raise CatalogUnavailable(f"No se pudo descargar la foto (HTTP {response.status_code}).")
        if len(response.content) > self._max_photo_bytes:
            raise InvalidPhoto("La foto supera el tamaño máximo permitido.")
        return response.content


def _snapshot(garment_id: GarmentId, garment: dict) -> GarmentSnapshot:
    """GarmentReadModel del Digital Closet -> lo que este servicio necesita."""
    subcategory = garment.get("subcategory")
    is_accessory = garment.get("category") == ACCESSORY_CATEGORY
    return GarmentSnapshot(
        garment_id=garment_id,
        subcategory=subcategory,
        category=AccessoryCategory.from_subcategory(subcategory) if is_accessory else None,
        image_version=(garment.get("image") or {}).get("version"),
    )


def _error_text(response: httpx.Response) -> str:
    """Mensaje del formato de error del Digital Closet: {"error": {"code", "message"}}."""
    try:
        error = response.json().get("error", {})
        return f"{error.get('code')}: {error.get('message')}"
    except (ValueError, AttributeError):
        return response.text[:200]
