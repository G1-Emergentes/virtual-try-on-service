"""
Validación de fotos con Gemini (Google Gemini API).

Antes de gastar GPU en la reconstrucción 3D, Gemini revisa la foto:
  - Qué accesorio es (lentes, gorra/sombrero u otro).
  - Desde qué ángulo se tomó.
  - Si sirve para generar un buen modelo 3D, y si no, qué debe corregir el usuario.
  - Tipo de cristal y acabado del marco (para elegir el material del modelo).

Usa salida estructurada: Gemini responde siempre con un JSON que sigue el
esquema PhotoCheck, así la API no depende de interpretar texto libre.

Si no hay API key o Gemini falla, la validación se omite (no bloquea el flujo).
"""
import io
import os
from typing import List, Literal, Optional

from PIL import Image
from pydantic import BaseModel, Field

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class PhotoCheck(BaseModel):
    category: Literal["glasses", "hat", "other"] = Field(
        description="glasses = lentes de cualquier tipo; hat = gorra, sombrero o gorro; other = cualquier otra cosa"
    )
    view: Literal["frontal", "three_quarter", "side", "top", "back", "other"] = Field(
        description="Ángulo desde el que se ve el accesorio"
    )
    usable: bool = Field(description="True si la foto cumple las reglas para su categoría")
    issues: List[str] = Field(description="Problemas detectados, en español, cortos (lista vacía si no hay)")
    message_es: str = Field(description="Instrucción breve en español para el usuario")
    lens_type: Literal["clear", "sun_gray", "sun_brown", "sun_green", "mirror", "none"] = Field(
        description="Solo lentes: clear = cristal transparente (de medida); sun_gray / sun_brown / "
                    "sun_green = cristal de sol oscuro de ese tono; mirror = cristal espejado; "
                    "none = no son lentes",
    )
    frame_material: Literal["glossy_plastic", "matte_plastic", "metal", "none"] = Field(
        description="Acabado principal del marco o del accesorio: glossy_plastic = acetato o plástico "
                    "brillante; matte_plastic = plástico mate, tela o similar; metal = metálico",
    )
    box_2d: List[int] = Field(
        description="Caja del accesorio principal [ymin, xmin, ymax, xmax], normalizada de 0 a 1000. "
                    "Solo el accesorio: sin el fondo ni otros objetos",
    )


PROMPT = """Eres el control de calidad de una app que convierte la foto de un accesorio en un modelo 3D
para probárselo en realidad aumentada. Analiza la imagen y responde según el esquema.

Reglas para que la foto sea usable:
- Debe haber UN solo accesorio, completo dentro de la foto y enfocado.
- No debe estar puesto en una persona ni sostenido con la mano tapándolo.
- Fondo preferentemente liso. Un fondo con muchos objetos hace la foto NO usable.
- Lentes (glasses): SOLO usables si se ven DE FRENTE (view = frontal), con las patillas abiertas.
- Gorras o sombreros (hat): usables de lado o en 3/4 (side o three_quarter), con la cámara a la
  altura del accesorio. Una foto tomada desde arriba (top) NO es usable.
- Cualquier otro accesorio (other): no soportado todavía, usable = false.

Además describe el accesorio aunque la foto no sea usable:
- lens_type: el tipo de cristal de los lentes (para otros accesorios, "none").
- frame_material: el acabado del marco (lentes) o del material principal (gorras: casi siempre matte_plastic).
- box_2d: la caja que encierra SOLO el accesorio principal (incluidas sus patillas o visera),
  sin objetos del fondo. Formato [ymin, xmin, ymax, xmax] de 0 a 1000.

En message_es escribe UNA instrucción clara y amable para el usuario. Si la foto es usable,
escribe "Foto válida". Si no, di exactamente qué cambiar (por ejemplo: "Toma la foto de
los lentes de frente, sobre una superficie lisa")."""


def _api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def is_enabled() -> bool:
    return bool(_api_key())


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=_api_key())
    return _client


def check_photo(image: Image.Image, timeout_s: float = 20.0) -> Optional[dict]:
    """
    Devuelve el resultado de PhotoCheck como dict, o None si la validación
    no está disponible (sin API key, sin red o error de Gemini).
    """
    if not is_enabled():
        return None
    try:
        from google.genai import types

        img = image.convert("RGB")
        img.thumbnail((1024, 1024))                 # menos datos, misma información
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)

        response = _get_client().models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PhotoCheck,
                temperature=0.0,
                http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
            ),
        )
        result = PhotoCheck.model_validate_json(response.text)
        return result.model_dump()
    except Exception as exc:                        # la validación nunca debe tumbar la API
        print(f"[mirage-ai-3d] Gemini no disponible: {exc}")
        return None
