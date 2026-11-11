"""Configuración del servicio, leída de variables de entorno (archivo .env)."""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    output_dir: str
    texture_resolution: int
    max_photo_mb: int
    # URL base del Digital Closet, p. ej. http://host.docker.internal:8081/api/v1
    # Vacía = MODO LOCAL (sin verificación de la prenda; la foto viaja en la petición).
    dc_api_url: str
    dc_timeout_s: float
    # Categorías que ya se pueden probar en RA (etiqueta AR_READY). Las demás: MODEL_3D.
    ar_categories: frozenset = frozenset({"glasses"})
    # Prefijo de todas las rutas del servicio. El API Gateway enruta por él
    # (p. ej. /api/v1/try-on/* -> este servicio), sin chocar con /api/v1/garments del Digital Closet.
    api_prefix: str = "/api/v1/try-on"
    # Ruta con la que la APP ve este servicio; con ella se arman model_url y thumbnail_url.
    # Sin gateway es igual al prefijo. Si el gateway publica otra ruta, se cambia aquí.
    public_base_path: str = "/api/v1/try-on"
    # Clave para /health/details (diagnóstico interno). Vacía = ese endpoint no existe (404).
    diagnostics_key: str = ""
    # Mostrar Swagger (/docs, /redoc, /openapi.json). En producción: false.
    expose_docs: bool = True

    @property
    def local_mode(self) -> bool:
        return not self.dc_api_url

    @classmethod
    def from_env(cls) -> "Settings":
        api_prefix = _normalize_path(os.getenv("API_PREFIX", "/api/v1/try-on"))
        return cls(
            output_dir=os.getenv("OUTPUT_DIR", "/outputs"),
            texture_resolution=int(os.getenv("TEXTURE_RESOLUTION", "1024")),
            max_photo_mb=int(os.getenv("MAX_PHOTO_MB", "10")),
            dc_api_url=os.getenv("DC_API_URL", "").strip(),
            dc_timeout_s=float(os.getenv("DC_TIMEOUT_S", "15")),
            ar_categories=frozenset(c.strip().lower() for c in os.getenv("AR_CATEGORIES", "glasses").split(",")
                                    if c.strip()),
            api_prefix=api_prefix,
            public_base_path=_normalize_path(os.getenv("PUBLIC_BASE_PATH", api_prefix)),
            diagnostics_key=os.getenv("DIAGNOSTICS_KEY", "").strip(),
            expose_docs=os.getenv("EXPOSE_DOCS", "true").strip().lower() in ("1", "true", "yes"),
        )


def _normalize_path(path: str) -> str:
    """'/api/v1/try-on/' -> '/api/v1/try-on'; '' -> '' (sin prefijo)."""
    path = path.strip().rstrip("/")
    if path and not path.startswith("/"):
        path = "/" + path
    return path
