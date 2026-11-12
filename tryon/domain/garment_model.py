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
- Un modelo listo tiene siempre la variante completa (FULL, para el visor 3D) y puede
  tener además la variante para RA (AR: sin patillas, para el probador). Las dos
  comparten origen y tamaño, así que usan el mismo ajuste en la cara (ArFit).
"""
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .errors import InvalidArFit, InvalidGarmentId, UnsupportedAccessory


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


class ModelVariant(str, Enum):
    """
    Versiones del .glb de un mismo modelo.
    - FULL: el modelo completo (con patillas). Lo muestra el visor 3D.
    - AR: sin patillas, cortado en las bisagras. Lo usa el probador de realidad aumentada:
      en la cara las patillas quedan detrás de la cabeza y se verían flotando.
    """

    FULL = "full"
    AR = "ar"


class ArFitSource(str, Enum):
    DEFAULT = "default"          # valor por defecto de la categoría (configurable)
    CALIBRATED = "calibrated"    # ajustado para este modelo desde la app


@dataclass(frozen=True)
class ArFit:
    """
    Cómo se coloca el modelo en la cara (valor inmutable).

    El .glb ya trae tamaño real y el origen en el punto de anclaje (lentes: centro del
    puente a la altura del centro de los cristales, plano frontal). Esto es el ajuste fino
    que aplica la app sobre ese contrato:
    - anchor: punto de la cara. Lentes -> "nose_bridge" (landmark 6 de MediaPipe).
    - offset_m: desplazamiento en metros en los ejes de la cara (x derecha, y arriba,
      z hacia fuera de la cara).
    - scale: factor sobre el tamaño real. pitch_deg: inclinación sobre el eje x.
    """

    anchor: str
    offset_m: tuple[float, float, float]
    scale: float = 1.0
    pitch_deg: float = 0.0
    source: ArFitSource = ArFitSource.DEFAULT

    # Límites: fuera de ellos el modelo ya no estaría en la cara.
    MAX_OFFSET_M = 0.03
    MIN_SCALE, MAX_SCALE = 0.5, 2.0
    MAX_PITCH_DEG = 30.0

    def __post_init__(self):
        if not self.anchor:
            raise InvalidArFit("Falta el punto de anclaje.")
        try:
            offset = tuple(float(v) for v in self.offset_m)
            scale, pitch = float(self.scale), float(self.pitch_deg)
        except (TypeError, ValueError):
            raise InvalidArFit("offset_m, scale y pitch_deg deben ser números.")
        if len(offset) != 3 or not all(math.isfinite(v) for v in (*offset, scale, pitch)):
            raise InvalidArFit("offset_m debe tener 3 números finitos (x, y, z) en metros.")
        if any(abs(v) > self.MAX_OFFSET_M for v in offset):
            raise InvalidArFit(f"Cada desplazamiento debe estar entre ±{self.MAX_OFFSET_M} m.")
        if not self.MIN_SCALE <= scale <= self.MAX_SCALE:
            raise InvalidArFit(f"La escala debe estar entre {self.MIN_SCALE} y {self.MAX_SCALE}.")
        if abs(pitch) > self.MAX_PITCH_DEG:
            raise InvalidArFit(f"La inclinación debe estar entre ±{self.MAX_PITCH_DEG}°.")
        object.__setattr__(self, "offset_m", offset)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "pitch_deg", pitch)

    @classmethod
    def from_dict(cls, data: dict, anchor: str,
                  source: ArFitSource = ArFitSource.DEFAULT) -> "ArFit":
        """El anclaje lo fija la categoría, no quien envía el ajuste."""
        if not isinstance(data, dict):
            raise InvalidArFit("El ajuste debe ser un objeto.")
        return cls(anchor=anchor, offset_m=data.get("offset_m", (0.0, 0.0, 0.0)),
                   scale=data.get("scale", 1.0), pitch_deg=data.get("pitch_deg", 0.0),
                   source=source)

    def to_dict(self) -> dict:
        return {"anchor": self.anchor, "offset_m": list(self.offset_m), "scale": self.scale,
                "pitch_deg": self.pitch_deg, "source": self.source.value}


# Punto de anclaje de cada categoría (coincide con el origen del .glb, ver postprocess.py).
# nose_bridge = landmark 6 de MediaPipe (puente nasal, entre los ojos).
AR_ANCHORS = {"glasses": "nose_bridge"}

# Ajuste por defecto de los lentes, calculado con el modelo canónico de MediaPipe (cm):
# landmark 6 = (0, 2.47, 5.79) y centro de los ojos en y = 2.62. Subir 1.5 mm pone los
# cristales a la altura de los ojos; adelantar 2 mm deja el frente en z ≈ 6.0, unos
# 12 mm delante de la córnea (distancia habitual). Escala 1: el .glb ya mide 14 cm.
DEFAULT_AR_FITS = {"glasses": {"offset_m": [0.0, 0.0015, 0.002], "scale": 1.0, "pitch_deg": 0.0}}


class ArSupportPolicy:
    """
    Qué tipos de accesorio ya se pueden probar en RA y cómo se colocan en la cara. Hoy
    solo lentes: es lo más configurado (anclaje en el puente nasal y tamaño real).
    Ampliar = agregar la categoría, su anclaje y su ajuste por defecto.
    """

    def __init__(self, ar_categories: frozenset = frozenset({"glasses"}),
                 default_fits: Optional[dict] = None):
        self._ar_categories = frozenset(ar_categories)
        fits = {**DEFAULT_AR_FITS, **(default_fits or {})}
        # Se valida al arrancar: un ajuste por defecto inválido es un error de configuración.
        self._default_fits = {
            category: ArFit.from_dict(fit, AR_ANCHORS[category])
            for category, fit in fits.items() if category in AR_ANCHORS
        }

    def supports_ar(self, category: "AccessoryCategory") -> bool:
        return category.value in self._ar_categories and category.value in self._default_fits

    def fit_for(self, model: "GarmentModel") -> Optional[ArFit]:
        """Ajuste de un modelo listo para RA: el calibrado si lo tiene, si no el de su categoría."""
        if model.badge(self) is not ModelBadge.AR_READY:
            return None
        calibrated = model.details.get("ar_fit")
        if calibrated is not None:
            try:
                return ArFit.from_dict(calibrated, AR_ANCHORS[model.category.value],
                                       ArFitSource.CALIBRATED)
            except InvalidArFit:
                pass    # dato antiguo o corrupto: se usa el de la categoría
        return self._default_fits[model.category.value]

    def calibrated_fit(self, model: "GarmentModel", data: dict) -> ArFit:
        """Valida el ajuste que envía la app para este modelo."""
        if model.badge(self) is not ModelBadge.AR_READY:
            raise UnsupportedAccessory("Este modelo no se puede probar en realidad aumentada.")
        return ArFit.from_dict(data, AR_ANCHORS[model.category.value], ArFitSource.CALIBRATED)


# Clave de details donde la generación describe la variante AR (cómo se cortó, etc.).
AR_VARIANT_DETAILS = "ar_variant"


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

    def has_variant(self, variant: ModelVariant) -> bool:
        """
        True si este modelo tiene esa variante del .glb. FULL existe en todo modelo listo;
        AR solo si la generación la creó (lentes generados desde que existe la variante).
        """
        if not self.is_ready:
            return False
        return variant is ModelVariant.FULL or AR_VARIANT_DETAILS in self.details

    def badge(self, policy: ArSupportPolicy) -> Optional[ModelBadge]:
        """Etiqueta de la tarjeta. Sin modelo listo no hay etiqueta."""
        if not self.is_ready:
            return None
        return ModelBadge.AR_READY if policy.supports_ar(self.category) else ModelBadge.MODEL_3D

    def calibrate(self, fit: ArFit, now: Optional[datetime] = None) -> None:
        """Guarda el ajuste en la cara de ESTE modelo. Regenerar lo descarta (otra geometría)."""
        self._require(ModelStatus.READY)
        stored = fit.to_dict()
        del stored["source"]
        self.details = {**self.details, "ar_fit": stored}
        self.updated_at = now or utc_now()

    def is_stale(self, current_image_version: Optional[int]) -> bool:
        """True si la prenda tiene ahora otra foto que la usada para generar este modelo."""
        if current_image_version is None or self.source_image_version is None:
            return False
        return current_image_version != self.source_image_version

    def _require(self, expected: ModelStatus) -> None:
        if self.status is not expected:
            raise ValueError(f"Transición inválida: el modelo está en {self.status.value}, "
                             f"se esperaba {expected.value}.")
