"""Dobles de prueba de los puertos del dominio."""
from pathlib import Path
from typing import Optional

from tryon.domain.errors import GarmentNotFound, GarmentWithoutPhoto, GenerationFailed
from tryon.domain.garment_model import AccessoryCategory, GarmentId, GarmentModel
from tryon.domain.ports import (Credentials, GarmentPhoto, GarmentSnapshot, GeneratedModel,
                                GenerationOptions, PhotoUpload)

GLASSES_ID = "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"


class FakeCatalog:
    def __init__(self, garments: dict[str, tuple[Optional[str], Optional[int]]]):
        # garment_id -> (subcategory, image_version)
        self.garments = garments

    def get_garment(self, garment_id: GarmentId, credentials: Credentials) -> GarmentSnapshot:
        if garment_id.value not in self.garments:
            raise GarmentNotFound("no existe")
        sub, version = self.garments[garment_id.value]
        return GarmentSnapshot(garment_id, sub, AccessoryCategory.from_subcategory(sub), version)

    def get_garments(self, garment_ids, credentials):
        result = {}
        for gid in garment_ids:
            try:
                result[gid] = self.get_garment(gid, credentials)
            except GarmentNotFound:
                result[gid] = None
        return result

    def get_photo(self, garment_id: GarmentId, credentials: Credentials,
                  upload: Optional[PhotoUpload]) -> GarmentPhoto:
        sub, version = self.garments[garment_id.value]
        if version is None:
            raise GarmentWithoutPhoto("sin foto")
        return GarmentPhoto(b"jpeg-bytes", version)


class InMemoryRepository:
    def __init__(self):
        self.models: dict[str, GarmentModel] = {}
        self.files: dict[str, Path] = {}
        self.thumbnails: dict[str, bytes] = {}
        self.photos: dict[str, GarmentPhoto] = {}
        self.history: list[str] = []          # estados guardados, en orden

    def get(self, garment_id):
        return self.models.get(garment_id.value)

    def save(self, model):
        self.models[model.garment_id.value] = model
        self.history.append(model.status.value)

    def store_photo(self, garment_id, photo):
        self.photos[garment_id.value] = photo

    def store_model_file(self, garment_id, generated_file):
        self.files[garment_id.value] = generated_file

    def model_file(self, garment_id):
        return self.files.get(garment_id.value)

    def store_thumbnail(self, garment_id, png):
        self.thumbnails[garment_id.value] = png

    def thumbnail_file(self, garment_id):
        return Path("thumb.png") if garment_id.value in self.thumbnails else None


class FakeGenerator:
    def __init__(self, tmp_dir: Path, fail: bool = False):
        self.tmp_dir = tmp_dir
        self.fail = fail
        self.calls: list[tuple[AccessoryCategory, GenerationOptions]] = []

    def generate(self, photo, category, options):
        self.calls.append((category, options))
        if self.fail:
            raise GenerationFailed("falló la GPU")
        path = self.tmp_dir / "job.glb"
        path.write_bytes(b"glTF-fake")
        return GeneratedModel(file_path=path, job_id="job123", details={"seconds": 1.0},
                              thumbnail_png=b"\x89PNG-fake")
