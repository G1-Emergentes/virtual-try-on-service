"""
Mirage - Virtual Try-On Service.

Genera el modelo 3D (.glb) de un accesorio a partir de su foto y lo asocia al ID de la
prenda en el Digital Closet. Este archivo solo arma la aplicación (raíz de composición):
la lógica vive en el paquete `tryon` (ver tryon/__init__.py y README.md).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tryon.bootstrap import build_container
from tryon.config import Settings
from tryon.infrastructure.generation.pipeline import AccessoryModelPipeline
from tryon.interfaces.http.errors import register_error_handlers
from tryon.interfaces.http.garment_model_router import router as garment_model_router
from tryon.interfaces.http.generation_router import health_router
from tryon.interfaces.http.generation_router import router as generation_router

# Se lee al importar: FastAPI necesita las rutas y la documentación antes de arrancar.
_SETTINGS = Settings.from_env()
API_PREFIX = _SETTINGS.api_prefix
DOCS = _SETTINGS.expose_docs          # EXPOSE_DOCS=false en producción oculta Swagger


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    pipeline = AccessoryModelPipeline.load(settings.output_dir)   # carga el modelo de IA (una vez)
    app.state.container = build_container(settings, pipeline)
    yield
    del app.state.container


app = FastAPI(
    title="Mirage - Virtual Try-On Service",
    version="0.4.0",
    description="Foto de un accesorio -> modelo 3D (.glb) asociado a la prenda del Digital Closet.",
    lifespan=lifespan,
    docs_url="/docs" if DOCS else None,
    redoc_url="/redoc" if DOCS else None,
    openapi_url="/openapi.json" if DOCS else None,
    # El token puesto en "Authorize" sobrevive a recargar la página de Swagger
    swagger_ui_parameters={"persistAuthorization": True},
    # Grupos de Swagger, uno por recurso (como en IAM: Users, Roles, Auth...)
    openapi_tags=[
        {"name": "Garment Models", "description": "Modelo 3D (.glb) de cada prenda, asociado a su ID del Digital Closet."},
        {"name": "Generations", "description": "Generación directa foto -> .glb, para pruebas del motor."},
        {"name": "Photo Checks", "description": "Validación de fotos con Gemini antes de generar."},
        {"name": "Health", "description": "¿Está vivo y listo el servicio? (sin datos internos)"},
    ],
)
register_error_handlers(app)
# Todas las rutas de negocio bajo el prefijo (p. ej. /api/v1/try-on/garments/...).
app.include_router(garment_model_router, prefix=API_PREFIX)
app.include_router(generation_router, prefix=API_PREFIX)
# /health queda también en la raíz: Docker, el gateway y los monitores lo consultan ahí.
app.include_router(health_router)
app.include_router(health_router, prefix=API_PREFIX, include_in_schema=False)
