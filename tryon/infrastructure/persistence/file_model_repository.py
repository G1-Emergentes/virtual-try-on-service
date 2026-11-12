"""
Adaptador del puerto ModelRepository sobre el disco.

Cada prenda tiene su carpeta, con el mismo ID del Digital Closet:

    {OUTPUT_DIR}/garments/{garment_id}/
        model.glb        el modelo 3D completo (visor 3D)
        model_ar.glb     variante para RA, sin patillas (probador; solo lentes)
        thumbnail.png    miniatura para la tarjeta: la foto sin fondo (PNG transparente)
        photo.jpg|png    la foto usada para generarlo
        metadata.json    estado, versión de la foto, fechas y datos de la generación

El modelo de dominio se convierte a JSON y de vuelta aquí, no en el dominio.
La configuración de Storage permite usar el adaptador de Supabase en su lugar.
"""
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from ...domain.garment_model import AccessoryCategory, GarmentId, GarmentModel, ModelStatus, ModelVariant
from ...domain.ports import GarmentPhoto

MODEL_FILE = "model.glb"
# Nombre del .glb de cada variante (lo comparten el disco y Supabase Storage).
MODEL_FILES = {ModelVariant.FULL: MODEL_FILE, ModelVariant.AR: "model_ar.glb"}
THUMBNAIL_FILE = "thumbnail.png"
METADATA_FILE = "metadata.json"
PHOTO_EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class FileModelRepository:

    def __init__(self, root_dir: str):
        self._root = Path(root_dir) / "garments"
        self._root.mkdir(parents=True, exist_ok=True)

    # --- Puerto ModelRepository ------------------------------------------------------------

    def get(self, garment_id: GarmentId) -> Optional[GarmentModel]:
        path = self._dir(garment_id) / METADATA_FILE
        if not path.exists():
            return None
        return _from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, model: GarmentModel) -> None:
        folder = self._dir(model.garment_id, create=True)
        _atomic_write(folder / METADATA_FILE, json.dumps(_to_dict(model), indent=2, ensure_ascii=False))

    def store_photo(self, garment_id: GarmentId, photo: GarmentPhoto) -> None:
        folder = self._dir(garment_id, create=True)
        for old in folder.glob("photo.*"):
            old.unlink()
        extension = PHOTO_EXTENSIONS.get(photo.content_type, ".jpg")
        (folder / f"photo{extension}").write_bytes(photo.data)

    def store_model_file(self, garment_id: GarmentId, generated_file: Path,
                         variant: ModelVariant = ModelVariant.FULL) -> None:
        target = self._dir(garment_id, create=True) / MODEL_FILES[variant]
        tmp = target.with_suffix(".glb.tmp")
        shutil.copyfile(generated_file, tmp)
        os.replace(tmp, target)                 # reemplazo atómico: nunca queda a medias

    def model_file(self, garment_id: GarmentId,
                   variant: ModelVariant = ModelVariant.FULL) -> Optional[Path]:
        path = self._dir(garment_id) / MODEL_FILES[variant]
        return path if path.exists() else None

    def store_thumbnail(self, garment_id: GarmentId, png: bytes) -> None:
        target = self._dir(garment_id, create=True) / THUMBNAIL_FILE
        tmp = target.with_suffix(".png.tmp")
        tmp.write_bytes(png)
        os.replace(tmp, target)

    def thumbnail_file(self, garment_id: GarmentId) -> Optional[Path]:
        path = self._dir(garment_id) / THUMBNAIL_FILE
        return path if path.exists() else None

    def has_thumbnail(self, garment_id: GarmentId) -> bool:
        return self.thumbnail_file(garment_id) is not None

    # --- Detalles --------------------------------------------------------------------------

    def _dir(self, garment_id: GarmentId, create: bool = False) -> Path:
        folder = self._root / garment_id.value      # GarmentId ya validó el UUID: ruta segura
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _to_dict(model: GarmentModel) -> dict:
    return {
        "garment_id": model.garment_id.value,
        "category": model.category.value,
        "status": model.status.value,
        "source_image_version": model.source_image_version,
        "job_id": model.job_id,
        "error": model.error,
        "created_at": model.created_at.isoformat(),
        "updated_at": model.updated_at.isoformat(),
        "details": model.details,
    }


def _from_dict(data: dict) -> GarmentModel:
    return GarmentModel(
        garment_id=GarmentId(data["garment_id"]),
        category=AccessoryCategory(data["category"]),
        status=ModelStatus(data["status"]),
        source_image_version=data.get("source_image_version"),
        created_at=datetime.fromisoformat(data["created_at"]),
        updated_at=datetime.fromisoformat(data["updated_at"]),
        job_id=data.get("job_id"),
        error=data.get("error"),
        details=data.get("details") or {},
    )
