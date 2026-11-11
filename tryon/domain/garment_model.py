"""
Agregado GarmentModel: el modelo 3D de una prenda del armario.

Vocabulario (lenguaje ubicuo):
- Prenda (garment): la gestiona el Digital Closet, que es su dueño y crea su ID.
- Accesorio: una prenda de categoría ACCESSORY. Aquí solo se generan lentes y gorras.
- Modelo 3D (GarmentModel): el .glb de UNA prenda. Se identifica con el mismo ID de la
  prenda, así la asociación prenda <-> modelo es directa y no necesita otra tabla.

Reglas:
- Una prenda tiene a lo sumo un modelo. Generar otra vez lo reemplaza.
- El modelo recuerda de qué versión de la foto salió (source_image_version). Si la prenda
  cambia de foto, el modelo queda desactualizado (is_stale) y conviene regenerarlo.
- Estados: PROCESSING -> READY | FAILED.
"""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .errors import InvalidGarmentId


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class GarmentId:
    """ID de la prenda, el mismo que genera el Digital Closet (UUID de 36 caracteres)."""

    value: str

    def __post_init__(self):
        try:
            canonical = str(uuid.UUID(self.value))
        except (ValueError, AttributeError, TypeError):
            raise InvalidGarmentId(f"'{self.value}' no es un ID de prenda válido (se espera un UUID).")
        if canonical != self.value.lower():
            raise InvalidGarmentId(f"'{self.value}' no es un ID de prenda válido (se espera un UUID).")
        object.__setattr__(self, "value", canonical)

    def __str__(self) -> str:
        return self.value


class AccessoryCategory(str, Enum):
    """Tipos de accesorio que el servicio sabe convertir en 3D."""

    GLASSES = "glasses"
    HAT = "hat"

    @classmethod
    def from_subcategory(cls, subcategory: Optional[str]) -> Optional["AccessoryCategory"]:
        """
        Traduce la subcategoría del Digital Closet (p. ej. "SUNGLASSES") a la categoría del
        motor 3D. Devuelve None si no se reconoce.
        """
        if not subcategory:
            return None
        return _SUBCATEGORY_TO_CATEGORY.get(subcategory.strip().upper())


# Subcategorías del Digital Closet (códigos ^[A-Z0-9_]+$) que se generan en 3D.
_SUBCATEGORY_TO_CATEGORY = {
    "GLASSES": AccessoryCategory.GLASSES,
    "EYEGLASSES": AccessoryCategory.GLASSES,
    "SUNGLASSES": AccessoryCategory.GLASSES,
    "HAT": AccessoryCategory.HAT,
    "CAP": AccessoryCategory.HAT,
    "BASEBALL_CAP": AccessoryCategory.HAT,
    "BUCKET_HAT": AccessoryCategory.HAT,
    "BEANIE": AccessoryCategory.HAT,
    "FEDORA": AccessoryCategory.HAT,
}


class ModelBadge(str, Enum):
    """
    Etiqueta que la app muestra en la tarjeta de la prenda.
    - AR_READY: hay modelo y se puede probar en la cara (realidad aumentada).
    - MODEL_3D: hay modelo, pero por ahora solo se puede ver en 3D.
    """

    AR_READY = "AR_READY"
    MODEL_3D = "MODEL_3D"


class ArSupportPolicy:
    """
    Qué tipos de accesorio ya se pueden probar en RA. Hoy solo lentes: es lo más
    configurado (anclaje en el puente nasal y tamaño real). Ampliar = agregar la categoría.
    """

    def __init__(self, ar_categories: frozenset = frozenset({"glasses"})):
        self._ar_categories = frozenset(ar_categories)

    def supports_ar(self, category: "AccessoryCategory") -> bool:
        return category.value in self._ar_categories


class ModelStatus(str, Enum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass
class GarmentModel:
    """Raíz del agregado. Solo cambia de estado a través de sus métodos."""

    garment_id: GarmentId
    category: AccessoryCategory
    status: ModelStatus
    source_image_version: Optional[int]
    created_at: datetime
    updated_at: datetime
    job_id: Optional[str] = None
    error: Optional[str] = None
    details: dict = field(default_factory=dict)

    # --- Fábrica ---------------------------------------------------------------------------

    @classmethod
    def start(cls, garment_id: GarmentId, category: AccessoryCategory,
              source_image_version: Optional[int], now: Optional[datetime] = None) -> "GarmentModel":
        """Abre una generación nueva para la prenda (reemplaza la anterior, si había)."""
        now = now or utc_now()
        return cls(garment_id=garment_id, category=category, status=ModelStatus.PROCESSING,
                   source_image_version=source_image_version, created_at=now, updated_at=now)

    # --- Transiciones ----------------------------------------------------------------------

    def complete(self, job_id: str, details: dict, now: Optional[datetime] = None) -> None:
        self._require(ModelStatus.PROCESSING)
        self.status = ModelStatus.READY
        self.job_id = job_id
        self.details = dict(details)
        self.error = None
        self.updated_at = now or utc_now()

    def fail(self, reason: str, now: Optional[datetime] = None) -> None:
        self._require(ModelStatus.PROCESSING)
        self.status = ModelStatus.FAILED
        self.error = reason
        self.updated_at = now or utc_now()

    # --- Consultas -------------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self.status is ModelStatus.READY

    def badge(self, policy: ArSupportPolicy) -> Optional[ModelBadge]:
        """Etiqueta de la tarjeta. Sin modelo listo no hay etiqueta."""
        if not self.is_ready:
            return None
        return ModelBadge.AR_READY if policy.supports_ar(self.category) else ModelBadge.MODEL_3D

    def is_stale(self, current_image_version: Optional[int]) -> bool:
        """True si la prenda tiene ahora otra foto que la usada para generar este modelo."""
        if current_image_version is None or self.source_image_version is None:
            return False
        return current_image_version != self.source_image_version

    def _require(self, expected: ModelStatus) -> None:
        if self.status is not expected:
            raise ValueError(f"Transición inválida: el modelo está en {self.status.value}, "
                             f"se esperaba {expected.value}.")
