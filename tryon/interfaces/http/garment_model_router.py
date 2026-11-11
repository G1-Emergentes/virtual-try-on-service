"""
API de modelos 3D por prenda: el modelo queda asociado al ID de la prenda del Digital Closet.

    POST /garments/{garment_id}/model             genera (o regenera) el modelo 3D
    GET  /garments/{garment_id}/model             descarga el .glb (pantalla de detalle)
    GET  /garments/{garment_id}/model/info        estado, etiqueta y datos del modelo
    GET  /garments/{garment_id}/model/thumbnail   miniatura: foto sin fondo (tarjeta)
    GET  /garments/models?ids=a,b,c               estado y etiqueta de varias prendas (armario)
    PUT  /garments/{garment_id}/model/ar-fit      guarda cómo se coloca en la cara (calibración)

Las rutas llevan el prefijo API_PREFIX (por defecto /api/v1/try-on). Las URLs que se
devuelven (model_url, thumbnail_url) usan PUBLIC_BASE_PATH: la ruta que ve la app.

Etiquetas (badge): AR_READY = se puede probar en la cara; MODEL_3D = solo ver en 3D;
null = sin modelo listo. Hoy solo los lentes son AR_READY (variable AR_CATEGORIES).

Autenticación: Authorization: Bearer <token de IAM>. En Swagger se pone una sola vez con el
botón "Authorize" (candado) y queda guardado. Este servicio no valida el token: lo reenvía al
Digital Closet, que lo valida contra IAM y comprueba que la prenda es del usuario.
En MODO LOCAL no se verifica nada.
"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ...application.generate_garment_model import GenerateGarmentModelCommand
from ...application.get_garment_model import GarmentModelSummary, GarmentModelView
from ...bootstrap import Container
from ...domain.garment_model import AccessoryCategory, ArFit, GarmentModel, ModelBadge
from ...domain.ports import Credentials, GenerationOptions, PhotoUpload

router = APIRouter(prefix="/garments", tags=["Garment Models"])


# --- Contrato de respuesta ---------------------------------------------------------------

class ArFitResponse(BaseModel):
    """Cómo se coloca el modelo en la cara. Solo en modelos AR_READY."""

    anchor: str                          # glasses -> nose_bridge (landmark 168 de MediaPipe)
    offset_m: list[float]                # x derecha, y arriba, z hacia fuera de la cara (metros)
    scale: float                         # sobre el tamaño real del .glb
    pitch_deg: float                     # inclinación sobre el eje x
    source: str                          # default | calibrated


class ArFitRequest(BaseModel):
    """Ajuste calibrado en la app. El anclaje lo fija la categoría del modelo."""

    offset_m: list[float] = Field(..., min_length=3, max_length=3,
                                  examples=[[0.0, 0.003, 0.006]])
    scale: float = Field(1.0, examples=[1.0])
    pitch_deg: float = Field(0.0, examples=[0.0])


class GarmentModelResponse(BaseModel):
    garment_id: str
    status: str                          # PROCESSING | READY | FAILED
    category: str                        # glasses | hat
    badge: Optional[str]                 # AR_READY | MODEL_3D | null
    source_image_version: Optional[int]  # versión de la foto usada
    is_stale: bool                       # la prenda cambió de foto: conviene regenerar
    model_url: Optional[str]             # ruta para descargar el .glb (si está READY)
    thumbnail_url: Optional[str]         # ruta de la miniatura (si existe)
    ar_fit: Optional[ArFitResponse]      # colocación en la cara (solo AR_READY)
    job_id: Optional[str]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    details: dict


class GarmentModelItem(BaseModel):
    """Fila de la consulta en lote: lo que necesita una tarjeta del armario."""

    garment_id: str
    found: bool                          # False = no existe o no es del usuario
    status: Optional[str]                # null = sin modelo
    category: Optional[str]
    badge: Optional[str]                 # AR_READY | MODEL_3D | null
    is_stale: bool
    model_url: Optional[str]
    thumbnail_url: Optional[str]
    ar_fit: Optional[ArFitResponse] = None


class GarmentModelList(BaseModel):
    items: list[GarmentModelItem]


class Links:
    """Arma las URLs que ve la app, con la ruta pública del servicio (PUBLIC_BASE_PATH)."""

    def __init__(self, base_path: str):
        self._base = base_path

    def model(self, garment_id: str) -> str:
        return f"{self._base}/garments/{garment_id}/model"

    def thumbnail(self, garment_id: str) -> str:
        return f"{self._base}/garments/{garment_id}/model/thumbnail"


def _badge(badge: Optional[ModelBadge]) -> Optional[str]:
    return badge.value if badge else None


def _ar_fit(fit: Optional[ArFit]) -> Optional[ArFitResponse]:
    return ArFitResponse(**fit.to_dict()) if fit else None


def _response(links: Links, model: GarmentModel, badge: Optional[ModelBadge], is_stale: bool = False,
              has_thumbnail: bool = False, ar_fit: Optional[ArFit] = None) -> GarmentModelResponse:
    gid = model.garment_id.value
    return GarmentModelResponse(
        garment_id=gid,
        status=model.status.value,
        category=model.category.value,
        badge=_badge(badge),
        source_image_version=model.source_image_version,
        is_stale=is_stale,
        model_url=links.model(gid) if model.is_ready else None,
        thumbnail_url=links.thumbnail(gid) if has_thumbnail else None,
        ar_fit=_ar_fit(ar_fit),
        job_id=model.job_id,
        error=model.error,
        created_at=model.created_at,
        updated_at=model.updated_at,
        details=model.details,
    )


def _item(links: Links, summary: GarmentModelSummary) -> GarmentModelItem:
    model = summary.model
    return GarmentModelItem(
        garment_id=summary.garment_id,
        found=summary.found,
        status=model.status.value if model else None,
        category=model.category.value if model else None,
        badge=_badge(summary.badge),
        is_stale=summary.is_stale,
        model_url=links.model(summary.garment_id) if model and model.is_ready else None,
        thumbnail_url=links.thumbnail(summary.garment_id) if summary.has_thumbnail else None,
        ar_fit=_ar_fit(summary.ar_fit),
    )


# --- Dependencias ------------------------------------------------------------------------

def _container(request: Request) -> Container:
    return request.app.state.container


def _links(request: Request) -> Links:
    return Links(_container(request).settings.public_base_path)


def _view_response(request: Request, view: GarmentModelView) -> GarmentModelResponse:
    return _response(_links(request), view.model, view.badge, view.is_stale, view.has_thumbnail,
                     view.ar_fit)


# Esquema de seguridad de OpenAPI: hace aparecer el candado "Authorize" en Swagger.
# auto_error=False: sin token no se corta aquí; decide el catálogo (en modo local no hace falta).
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="IAM token",
                           description="Token de acceso de IAM (el mismo que usa el Digital Closet).")


def credentials(token: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Credentials:
    """Token del usuario, leído del header Authorization: Bearer ..."""
    return Credentials(bearer_token=token.credentials if token else None)


# --- Endpoints ---------------------------------------------------------------------------

@router.get("/models", response_model=GarmentModelList,
            summary="Estado y etiqueta de varias prendas (pantalla del armario)")
async def list_garment_models(
    request: Request,
    ids: str = Query(..., description="IDs de prenda separados por coma (máximo 100)"),
    creds: Credentials = Depends(credentials),
):
    """
    Una sola llamada para pintar el armario: por cada prenda, si tiene modelo, su etiqueta
    (AR_READY o MODEL_3D) y las rutas del modelo y la miniatura. Respeta el orden pedido.
    """
    garment_ids = [i.strip() for i in ids.split(",") if i.strip()]
    summaries = await run_in_threadpool(_container(request).get_garment_model.many, garment_ids,
                                        creds)
    links = _links(request)
    return GarmentModelList(items=[_item(links, s) for s in summaries])


@router.post("/{garment_id}/model", response_model=GarmentModelResponse, status_code=201,
             summary="Generar el modelo 3D de una prenda")
async def generate_garment_model(
    request: Request,
    garment_id: str,
    creds: Credentials = Depends(credentials),
    category: Optional[AccessoryCategory] = Query(
        None, description="Opcional. Por defecto se deduce de la subcategoría del Digital Closet "
                          "(GLASSES, SUNGLASSES -> glasses; HAT, CAP -> hat). En modo local es obligatoria."),
    lens_type: Optional[str] = Query(None, pattern="^(clear|sun_gray|sun_brown|sun_green|mirror)$",
                                     description="Opcional. Por defecto se mide en la foto."),
    frame_material: Optional[str] = Query(None, pattern="^(glossy_plastic|matte_plastic|metal)$",
                                          description="Opcional. Acabado del marco."),
    image: Optional[UploadFile] = File(None, description="Solo MODO LOCAL: la foto del accesorio. "
                                                         "Con Digital Closet la foto se toma del armario."),
):
    """
    Genera el .glb de la prenda y lo guarda con su ID. Si ya tenía modelo, lo reemplaza.
    Tarda unos segundos (usa la GPU); la respuesta llega cuando el modelo está listo.
    """
    upload = None
    if image is not None:
        upload = PhotoUpload(data=await image.read(), content_type=image.content_type or "")
    command = GenerateGarmentModelCommand(
        garment_id=garment_id,
        credentials=creds,
        category=category,
        options=GenerationOptions(lens_type=lens_type, frame_material=frame_material),
        upload=upload,
    )
    container = _container(request)
    model = await run_in_threadpool(container.generate_garment_model.execute, command)
    view = await run_in_threadpool(container.get_garment_model.info, model.garment_id.value, command.credentials)
    return _view_response(request, view)


@router.get("/{garment_id}/model", summary="Descargar el .glb de una prenda",
            response_class=FileResponse,
            responses={200: {"content": {"model/gltf-binary": {}}}})
async def download_garment_model(
    request: Request,
    garment_id: str,
    creds: Credentials = Depends(credentials),
):
    view: GarmentModelView = await run_in_threadpool(
        _container(request).get_garment_model.file, garment_id, creds)
    model = view.model
    filename = f"{model.garment_id.value}.glb"
    headers = {
        "X-Garment-Id": model.garment_id.value,
        "X-Model-Status": model.status.value,
        "X-Source-Image-Version": str(model.source_image_version or ""),
        "X-Model-Stale": "yes" if view.is_stale else "no",
        "Cache-Control": "private, no-store",
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    if isinstance(view.file_path, bytes):
        return Response(view.file_path, media_type="model/gltf-binary", headers=headers)
    return FileResponse(view.file_path, media_type="model/gltf-binary", headers=headers)


@router.get("/{garment_id}/model/info", response_model=GarmentModelResponse,
            summary="Estado del modelo 3D de una prenda")
async def garment_model_info(
    request: Request,
    garment_id: str,
    creds: Credentials = Depends(credentials),
):
    view = await run_in_threadpool(_container(request).get_garment_model.info, garment_id,
                                   creds)
    return _view_response(request, view)


@router.get("/{garment_id}/model/thumbnail", summary="Miniatura de la prenda (foto sin fondo)",
            response_class=FileResponse, responses={200: {"content": {"image/png": {}}}})
async def garment_model_thumbnail(
    request: Request,
    garment_id: str,
    creds: Credentials = Depends(credentials),
):
    """PNG de 512 x 512 con fondo transparente: la app le pone el color de la tarjeta."""
    view = await run_in_threadpool(_container(request).get_garment_model.thumbnail, garment_id,
                                   creds)
    headers = {"X-Garment-Id": view.model.garment_id.value, "Cache-Control": "private, no-store"}
    if isinstance(view.thumbnail_path, bytes):
        return Response(view.thumbnail_path, media_type="image/png", headers=headers)
    return FileResponse(view.thumbnail_path, media_type="image/png", headers=headers)


@router.put("/{garment_id}/model/ar-fit", response_model=GarmentModelResponse,
            summary="Guardar cómo se coloca el modelo en la cara (calibración)")
async def calibrate_garment_model_ar_fit(
    request: Request,
    garment_id: str,
    fit: ArFitRequest,
    creds: Credentials = Depends(credentials),
):
    """
    Guarda el ajuste calibrado en la app para ESTE modelo (solo AR_READY). Desde entonces
    `ar_fit.source` es `calibrated`. Regenerar el modelo descarta la calibración.
    Límites: desplazamientos de ±0.03 m, escala de 0.5 a 2 e inclinación de ±30°.
    """
    view = await run_in_threadpool(_container(request).calibrate_ar_fit.execute, garment_id, creds,
                                   fit.model_dump())
    return _view_response(request, view)
