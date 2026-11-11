"""
Raíz de composición: el único lugar que conoce todas las piezas concretas y las conecta.
Los casos de uso reciben puertos; aquí se decide qué adaptador implementa cada uno.
"""
from dataclasses import dataclass

from .application.generate_garment_model import GenerateGarmentModel
from .application.get_garment_model import GetGarmentModel
from .config import Settings
from .domain.garment_model import ArSupportPolicy
from .domain.ports import GarmentCatalog
from .infrastructure.catalog.digital_closet_catalog import DigitalClosetCatalog
from .infrastructure.generation.pipeline import AccessoryModelPipeline
from .infrastructure.generation.spar3d_generator import Spar3dModelGenerator
from .infrastructure.persistence.file_model_repository import FileModelRepository


@dataclass(frozen=True)
class Container:
    settings: Settings
    pipeline: AccessoryModelPipeline
    generate_garment_model: GenerateGarmentModel
    get_garment_model: GetGarmentModel


def build_catalog(settings: Settings) -> GarmentCatalog:
    if not settings.local_mode:
        print(f"[mirage-ai-3d] Catálogo: Digital Closet en {settings.dc_api_url}")
        return DigitalClosetCatalog(settings.dc_api_url, settings.dc_timeout_s, settings.max_photo_mb)
    else:
        # MODO LOCAL (temporal). Para quitarlo: borrar esta rama y local_garment_catalog.py
        from .infrastructure.catalog.local_garment_catalog import LocalGarmentCatalog
        print("[mirage-ai-3d] Catálogo: MODO LOCAL (DC_API_URL vacía). La prenda no se verifica.")
        return LocalGarmentCatalog(settings.max_photo_mb)


def build_container(settings: Settings, pipeline: AccessoryModelPipeline) -> Container:
    catalog = build_catalog(settings)
    repository = FileModelRepository(settings.output_dir)
    generator = Spar3dModelGenerator(pipeline, settings.texture_resolution)
    return Container(
        settings=settings,
        pipeline=pipeline,
        generate_garment_model=GenerateGarmentModel(catalog, repository, generator),
        get_garment_model=GetGarmentModel(catalog, repository, ArSupportPolicy(settings.ar_categories)),
    )
