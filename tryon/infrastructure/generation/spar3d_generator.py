"""
Adaptador del puerto ModelGenerator sobre el motor 3D (SPAR3D o SF3D + posprocesado).
Traduce los errores técnicos (imagen ilegible, falta de VRAM) a errores del dominio.
"""
import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from ...domain.errors import GenerationFailed, InsufficientGpuMemory, InvalidPhoto
from ...domain.garment_model import AccessoryCategory
from ...domain.ports import GarmentPhoto, GeneratedModel, GenerationOptions
from .pipeline import AccessoryModelPipeline, PipelineParams
from .thumbnail import make_thumbnail


class Spar3dModelGenerator:

    def __init__(self, pipeline: AccessoryModelPipeline, texture_resolution: int = 1024):
        self._pipeline = pipeline
        self._texture_resolution = texture_resolution

    def generate(self, photo: GarmentPhoto, category: AccessoryCategory,
                 options: GenerationOptions) -> GeneratedModel:
        try:
            image = Image.open(io.BytesIO(photo.data))
            image.load()
        except (UnidentifiedImageError, OSError):
            raise InvalidPhoto("No se pudo leer la foto de la prenda.")

        params = PipelineParams(
            category=category.value,
            texture_resolution=self._texture_resolution,
            lens_type=options.lens_type,
            frame_material=options.frame_material,
        )
        try:
            path, meta, cutout = self._pipeline.run_with_cutout(image, params)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise InsufficientGpuMemory("Sin memoria de GPU suficiente. Cierra otras apps que usen la GPU.")
        except Exception as exc:
            raise GenerationFailed(f"No se pudo generar el modelo 3D: {exc}")

        ar = dict(meta.get("ar_variant") or {})
        ar_path = ar.pop("path", None)              # la ruta es interna; el resto va a details
        return GeneratedModel(file_path=Path(path), job_id=meta["job_id"], details=_summary(meta),
                              thumbnail_png=_safe_thumbnail(cutout),
                              ar_file_path=Path(ar_path) if ar_path else None, ar_details=ar)


def _safe_thumbnail(cutout):
    """La miniatura es un extra: si falla, el modelo se entrega igual."""
    try:
        return make_thumbnail(cutout)
    except Exception as exc:
        print(f"[mirage-ai-3d] miniatura omitida: {exc}")
        return None


def _summary(meta: dict) -> dict:
    """Solo los datos útiles para la app y para depurar (sin estructuras internas)."""
    lens = meta.get("lens_tint") or {}
    return {
        "model": meta.get("model"),
        "seconds": meta.get("seconds"),
        "peak_vram_mb": meta.get("peak_vram_mb"),
        "remesh": meta.get("remesh"),
        "postprocess": meta.get("mode"),
        "lens_type": lens.get("preset"),
        "frame_material": meta.get("frame_material"),
        "size_m": meta.get("size_m"),
        "anchor": meta.get("anchor"),               # lens_line | bridge (lentes)
    }
