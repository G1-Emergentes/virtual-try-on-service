"""
Endpoints técnicos (pruebas y diagnóstico), sin asociación a una prenda:

    GET  /health    vivo/listo, sin datos internos (también en la raíz)
    GET  /health/details  diagnóstico interno, solo con X-Diagnostics-Key (oculto en Swagger)
    POST /generate  foto -> .glb con todos los parámetros del motor (para experimentar)
    POST /check     valida una foto con Gemini, sin GPU

Las rutas llevan el prefijo API_PREFIX (por defecto /api/v1/try-on).
Para la app usa /garments/{garment_id}/model: ese sí queda asociado a la prenda.
"""
import hmac
import io
import os
from typing import Optional

import torch
from fastapi import APIRouter, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image

import gemini_check
from ...bootstrap import Container
from ...infrastructure.generation.pipeline import PipelineParams

router = APIRouter()
health_router = APIRouter()

# Alcance actual: lentes y gorras/sombreros. "auto" deja que Gemini lo detecte.
CATEGORY_PATTERN = "^(auto|glasses|hat|raw)$"
ALLOWED_TYPES = ("image/jpeg", "image/png", "image/webp")


def _container(request: Request) -> Container:
    return request.app.state.container


@health_router.get("/health", tags=["Health"], summary="Estado del servicio")
def health(request: Request):
    """
    Respuesta pública y mínima, para Docker, el gateway y los monitores.
    No expone versiones, hardware ni configuración (eso ayudaría a un atacante).
    """
    ready = getattr(request.app.state, "container", None) is not None
    return {"status": "ok" if ready else "starting"}


@health_router.get("/health/details", include_in_schema=False)
def health_details(request: Request, x_diagnostics_key: Optional[str] = Header(None)):
    """
    Diagnóstico interno (GPU, VRAM, modelo, catálogo). Solo existe si DIAGNOSTICS_KEY está
    configurada, y exige el header X-Diagnostics-Key con ese valor. Si no, responde 404.
    """
    container: Container = getattr(request.app.state, "container", None)
    key = container.settings.diagnostics_key if container else ""
    if not key or not x_diagnostics_key or not hmac.compare_digest(x_diagnostics_key, key):
        raise HTTPException(404, "Not Found")
    info = {"status": "ok", "gemini": gemini_check.is_enabled(),
            "device": container.pipeline.device, "model": container.pipeline.model_name,
            "catalog": "local" if container.settings.local_mode else "digital_closet"}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_free_mb"] = round(free / 1024**2)
        info["vram_total_mb"] = round(total / 1024**2)
    return info


@router.post("/generate", tags=["Generations"], summary="Foto -> .glb con todos los parámetros (sin prenda asociada)")
async def generate(
    request: Request,
    file: UploadFile = File(..., description="Foto del accesorio (JPG, PNG o WebP)"),
    texture_resolution: int = Query(int(os.getenv("TEXTURE_RESOLUTION", "1024")), ge=256, le=2048),
    remesh: str = Query("auto", pattern="^(auto|none|triangle|quad)$",
                        description="auto (quad con SPAR3D: malla ordenada y lisa) | none | triangle | quad"),
    category: str = Query("auto", pattern=CATEGORY_PATTERN,
                          description="auto (lo detecta Gemini) | glasses | hat | raw (sin posprocesado)"),
    validate: bool = Query(True, description="Revisar la foto con Gemini antes de generar"),
    force: bool = Query(False, description="Generar aunque Gemini indique que la foto no sirve"),
    postprocess: str = Query("auto", pattern="^(auto|light|full)$",
                             description="auto (light con SPAR3D, full con SF3D) | light (respeta la malla) | full (corrige defectos de SF3D)"),
    lens_type: str = Query("auto", pattern="^(auto|clear|sun_gray|sun_brown|sun_green|mirror)$",
                           description="Luna: auto (Gemini o medida en la foto) | clear | sun_gray | sun_brown | sun_green | mirror"),
    frame_material: str = Query("auto", pattern="^(auto|glossy_plastic|matte_plastic|metal|none)$",
                                description="Acabado del marco: auto | glossy_plastic | matte_plastic | metal | none (sin textura)"),
    keep_temples: bool = Query(False, description="Solo lentes, modo full: conservar patillas completas (en light siempre se conservan)"),
    keep_normal_map: bool = Query(False, description="Conservar el mapa de normales de SF3D (suele ser ruido)"),
    project_photo: bool = Query(True, description="Lentes, modo full: pegar la foto real sobre el frente del marco"),
    unlit: bool = Query(False, description="Material sin iluminación: colores idénticos a la foto, sin reflejos"),
):
    container = _container(request)
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Formato no soportado. Usa JPG, PNG o WebP.")
    data = await file.read()
    if len(data) > container.settings.max_photo_mb * 1024 * 1024:
        raise HTTPException(413, f"La imagen supera {container.settings.max_photo_mb} MB.")
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "No se pudo leer la imagen.")

    category, photo_check = await _resolve_category(image, category, validate, force)
    params = PipelineParams(
        category=category,
        texture_resolution=texture_resolution,
        remesh=remesh,
        postprocess=postprocess,
        keep_temples=keep_temples,
        keep_normal_map=keep_normal_map,
        project_photo=project_photo,
        unlit=unlit,
        lens_type=_from_check(lens_type, photo_check, "lens_type"),
        frame_material=_frame_material(frame_material, photo_check),
        box_2d=(photo_check or {}).get("box_2d"),
    )
    try:
        path, meta = await run_in_threadpool(container.pipeline.run, image, params)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(507, "Sin VRAM suficiente. Cierra otras apps que usen la GPU, "
                                 "prueba texture_resolution=512 o usa RECON_MODEL=sf3d.")

    lens = meta.get("lens_tint")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename=f"{meta['job_id']}.glb",
        headers={
            "X-Job-Id": meta["job_id"],
            "X-Inference-Seconds": str(meta["seconds"]),
            "X-Peak-VRAM-MB": str(meta["peak_vram_mb"]),
            "X-Category": category,
            "X-Model": meta.get("model", ""),
            "X-Remesh": meta.get("remesh", ""),
            "X-Postprocess": meta.get("mode", "full" if category != "raw" else "none"),
            "X-Lens-Type": f'{lens.get("preset", "photo")} ({lens.get("source", "photo")})' if lens else "default",
            "X-Cropped": "yes" if meta.get("cropped_to_box") else "no",
            "X-Frame-Material": str(meta.get("frame_material", "none")),
            "X-Size-M": ",".join(str(v) for v in meta.get("size_m", [])),
            "X-Quality": meta.get("quality", "ok"),
            "X-Photo-Check": "skipped" if photo_check is None else ("ok" if photo_check["usable"] else "forced"),
        },
    )


@router.post("/check", tags=["Photo Checks"], summary="Validar una foto con Gemini (sin GPU)")
async def check(file: UploadFile = File(..., description="Foto a revisar antes de generar")):
    """Solo valida la foto con Gemini (rápido, sin GPU). Útil para la cámara de la app."""
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "No se pudo leer la imagen.")
    result = await run_in_threadpool(gemini_check.check_photo, image)
    if result is None:
        raise HTTPException(503, "Gemini no está disponible (revisa GEMINI_API_KEY en .env).")
    return result


# --- Apoyo -------------------------------------------------------------------------------

async def _resolve_category(image, category, validate, force):
    """Valida la foto con Gemini y, si category=auto, usa la categoría que detecta."""
    check = None
    if validate:
        check = await run_in_threadpool(gemini_check.check_photo, image)
    if check is not None:
        if category == "auto":
            if check["category"] == "other":
                raise HTTPException(422, {
                    "error": "accesorio_no_soportado",
                    "message_es": "Por ahora solo se pueden generar lentes y gorras o sombreros.",
                    "photo_check": check,
                })
            category = check["category"]
        if not check["usable"] and not force:
            raise HTTPException(422, {
                "error": "foto_no_valida",
                "message_es": check["message_es"],
                "photo_check": check,
            })
    if category == "auto":
        raise HTTPException(400, "Gemini no está disponible: indica category=glasses o category=hat.")
    return category, check


def _from_check(value: str, check, key: str):
    """'auto' = lo que detectó Gemini (o nada, y el motor decide)."""
    if value != "auto":
        return value
    detected = (check or {}).get(key)
    return None if detected == "none" else detected


def _frame_material(value: str, check):
    if value == "none":
        return "off"                       # sin textura de la biblioteca
    return _from_check(value, check, "frame_material")
