"""
Motor 3D: foto -> .glb.

Envuelve los módulos del motor (recon.py, postprocess.py, projection.py,
material_presets.py) en una sola clase. La usan:
- el endpoint técnico POST /generate (pruebas con todos los parámetros);
- el adaptador Spar3dModelGenerator (puerto ModelGenerator del dominio).

Pasos: recorte opcional -> quitar fondo -> reconstrucción (GPU) -> posprocesado -> .glb.
Para lentes se exporta además la variante para RA (sin patillas): meta["ar_variant"].
"""
import os
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import trimesh
from PIL import Image

from postprocess import GlassesConfig, PostprocessOptions, export_glb, make_ar_variant, process_accessory
from recon import Reconstructor


@dataclass(frozen=True)
class PipelineParams:
    """Parámetros del motor. Los valores por defecto son los recomendados."""

    category: str                         # glasses | hat | raw (sin posprocesado)
    texture_resolution: int = 1024
    remesh: str = "auto"                  # auto = quad con SPAR3D (superficie más lisa)
    postprocess: str = "auto"             # auto = light con SPAR3D, full con SF3D
    keep_temples: bool = False
    keep_normal_map: bool = False
    project_photo: bool = True
    unlit: bool = False
    lens_type: Optional[str] = None
    frame_material: Optional[str] = None  # "off" = sin textura de la biblioteca
    box_2d: Optional[list] = None         # caja de Gemini para recortar el accesorio


class AccessoryModelPipeline:
    """Carga el modelo de IA una vez y genera .glb bajo demanda (una inferencia a la vez)."""

    def __init__(self, reconstructor: Reconstructor, device: str, output_dir: str):
        self._recon = reconstructor
        self.device = device
        self.output_dir = output_dir
        # Con 8 GB de VRAM solo cabe una inferencia a la vez
        self._gpu_lock = threading.Lock()

    @classmethod
    def load(cls, output_dir: str) -> "AccessoryModelPipeline":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[mirage-ai-3d] Dispositivo: {device}")
        recon = Reconstructor(device)
        os.makedirs(output_dir, exist_ok=True)
        print(f"[mirage-ai-3d] Modelo cargado: {recon.name} (bajo consumo de VRAM: {recon.low_vram})")
        return cls(recon, device, output_dir)

    @property
    def model_name(self) -> str:
        return self._recon.name

    def run(self, image: Image.Image, params: PipelineParams) -> tuple[str, dict]:
        """Genera el .glb. Devuelve (ruta, metadatos). Lanza torch.cuda.OutOfMemoryError."""
        path, meta, _ = self.run_with_cutout(image, params)
        return path, meta

    def run_with_cutout(self, image: Image.Image, params: PipelineParams) -> tuple[str, dict, Image.Image]:
        """
        Igual que run(), y además devuelve la foto sin fondo (RGBA) para la miniatura.

        Para lentes, meta["ar_variant"] trae la variante para el probador de RA:
        {"path": .glb sin patillas, "method": "hinge" | "crop_temples", ...}.
        Si no se pudo crear, no aparece (el probador usa el modelo completo).
        """
        image, cropped = crop_to_box(image, params.box_2d)
        original = image.convert("RGB")
        cutout = self._recon.cutout(image)
        prepared = self._recon.prepare(cutout)
        remesh = params.remesh
        if remesh == "auto":
            remesh = "quad" if self._recon.name == "spar3d" else "none"

        mesh, elapsed, peak_mb = self._reconstruct(prepared, params.texture_resolution, remesh)

        job_id = uuid.uuid4().hex
        raw_path = os.path.join(self.output_dir, f"{job_id}_raw.glb")
        mesh.export(raw_path, include_normals=True)
        meta = {"job_id": job_id, "seconds": round(elapsed, 2), "peak_vram_mb": round(peak_mb),
                "model": self._recon.name, "remesh": remesh}
        if params.category == "raw":
            return raw_path, meta, cutout

        mode = params.postprocess
        if mode == "auto":
            # SPAR3D entrega la malla completa: solo retoque ligero. SF3D necesita las correcciones.
            mode = "light" if self._recon.name == "spar3d" else "full"
        opts = PostprocessOptions(
            category=params.category,
            mode=mode,
            lens_type=params.lens_type,
            frame_material=params.frame_material,
            keep_normal_map=params.keep_normal_map,
            project_photo=params.project_photo,
            glasses=GlassesConfig(keep_temples=params.keep_temples),
        )
        scene, stats = process_accessory(trimesh.load(raw_path), cutout, opts, photo_original=original)
        path = os.path.join(self.output_dir, f"{job_id}.glb")
        # plástico: menos reflejo del entorno (el negro no se ve gris); metal: reflejo completo
        specular = 0.5 if stats.get("frame_material") not in (None, "none", "metal") else None
        export_glb(scene, path, unlit=params.unlit, specular=specular)
        meta["cropped_to_box"] = cropped
        meta.update(stats)
        if params.category == "glasses":
            meta.update(self._export_ar_variant(scene, opts, job_id, params.unlit, specular))
        return path, meta, cutout

    def _export_ar_variant(self, scene, opts: PostprocessOptions, job_id: str,
                           unlit: bool, specular) -> dict:
        """La variante para RA es un extra: si falla, el modelo completo se entrega igual."""
        try:
            ar_scene, ar_stats = make_ar_variant(scene, opts.glasses)
            ar_path = os.path.join(self.output_dir, f"{job_id}_ar.glb")
            export_glb(ar_scene, ar_path, unlit=unlit, specular=specular)
        except Exception as exc:
            print(f"[mirage-ai-3d] variante para RA omitida: {exc}")
            return {}
        return {"ar_variant": {"path": ar_path, **ar_stats}}

    def _reconstruct(self, image, texture_resolution: int, remesh: str):
        with self._gpu_lock:
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if self.device == "cuda" else nullcontext())
            with torch.no_grad(), autocast:
                mesh = self._recon.run(image, texture_resolution, remesh)
            elapsed = time.perf_counter() - start
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2 if self.device == "cuda" else 0
        return mesh, elapsed, peak_mb


def crop_to_box(image: Image.Image, box_2d, margin: float = 0.10):
    """
    Recorta la foto a la caja del accesorio que marcó Gemini (+10 % de margen).
    Así el quitafondos y la reconstrucción no toman objetos del fondo.
    """
    if not box_2d or len(box_2d) != 4:
        return image, False
    w, h = image.size
    y0, x0, y1, x1 = [max(0, min(1000, int(v))) for v in box_2d]
    if y1 - y0 < 50 or x1 - x0 < 50:                  # caja inválida o diminuta
        return image, False
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    box = (int(max(0, (x0 - mx) / 1000 * w)), int(max(0, (y0 - my) / 1000 * h)),
           int(min(w, (x1 + mx) / 1000 * w)), int(min(h, (y1 + my) / 1000 * h)))
    return image.crop(box), True
