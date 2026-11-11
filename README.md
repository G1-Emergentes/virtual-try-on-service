# Mirage — Virtual Try-On Service

Microservicio que convierte la foto de un accesorio en un modelo 3D (`.glb`) para probárselo en realidad aumentada. Cada modelo queda **asociado al ID de la prenda** del Digital Closet Service.

- Modelo de IA: **SPAR3D** de Stability AI (respaldo: SF3D). Corre en la GPU, dentro de Docker.
- Alcance actual: **lentes** y **gorras o sombreros**.
- Salida: `.glb` (glTF 2.0 binario), a tamaño real y con el origen en su punto de anclaje para la RA.

## Arquitectura

Arquitectura limpia con diseño guiado por el dominio (DDD). Las dependencias apuntan hacia dentro: `interfaces → application → domain`. La infraestructura implementa los puertos que define el dominio.

```
app.py                         Raíz de composición: arma FastAPI y conecta las piezas
tryon/
  config.py                    Configuración desde variables de entorno
  bootstrap.py                 Elige qué adaptador implementa cada puerto
  domain/                      Reglas del negocio (sin FastAPI, sin GPU, sin HTTP)
    garment_model.py           Agregado GarmentModel, GarmentId, AccessoryCategory, ModelStatus,
                               ModelBadge y ArSupportPolicy (qué categorías se prueban en RA)
    ports.py                   Puertos: GarmentCatalog, ModelRepository, ModelGenerator
    errors.py                  Errores del dominio con código estable
  application/                 Casos de uso
    generate_garment_model.py  Generar el modelo de una prenda
    get_garment_model.py       Consultar o descargar el modelo de una prenda
  infrastructure/              Adaptadores
    catalog/digital_closet_catalog.py   GarmentCatalog sobre la API del Digital Closet
    catalog/local_garment_catalog.py    GarmentCatalog del MODO LOCAL (temporal)
    persistence/file_model_repository.py  ModelRepository en disco
    generation/pipeline.py              Motor 3D: quitar fondo, SPAR3D, posprocesado
    generation/spar3d_generator.py      ModelGenerator sobre el motor 3D
    generation/thumbnail.py             Miniatura: foto sin fondo, cuadrada y centrada
  interfaces/http/             Entrada HTTP
    garment_model_router.py    /garments/{garment_id}/model...
    generation_router.py       /health, /generate, /check (técnicos)
    errors.py                  Error del dominio → código HTTP
recon.py, postprocess.py,      Motor 3D (detalle de infraestructura que usa pipeline.py)
projection.py, material_presets.py, gemini_check.py
materials/                     Biblioteca de materiales (presets + texturas CC0)
tests/                         Pruebas sin GPU (dobles de los puertos)
```

**Lenguaje del dominio**
- **Prenda (garment):** la gestiona el Digital Closet, que es su dueño y crea su ID.
- **Accesorio:** una prenda de categoría `ACCESSORY`. Subcategorías que se generan: `GLASSES`, `EYEGLASSES`, `SUNGLASSES` → lentes; `HAT`, `CAP`, `BASEBALL_CAP`, `BUCKET_HAT`, `BEANIE`, `FEDORA` → gorras o sombreros.
- **Modelo 3D (GarmentModel):** el `.glb` de una prenda, con su mismo ID. Generar otra vez lo reemplaza. Guarda la versión de la foto con la que se generó: si la prenda cambia de foto, queda desactualizado (`is_stale`).
- **Estados:** `PROCESSING → READY | FAILED`.

## Integración con el Digital Closet

Este servicio no modifica el Digital Closet: usa endpoints que ya tiene y reenvía el token del usuario, como su `visualization-worker`.

1. La app crea la prenda y sube su foto en el Digital Closet. Así obtiene el `garment_id`.
2. La app llama a `POST /garments/{garment_id}/model` en este servicio, con `Authorization: Bearer <token de IAM>`.
3. Este servicio consulta al Digital Closet:
   - `GET /garments/{id}`: la prenda existe, es del usuario y es un accesorio soportado;
   - `GET /garments/{id}/image-access`: la URL firmada de la foto, que descarga.
4. Genera el `.glb` y lo guarda en `outputs/garments/{garment_id}/`.
5. La app descarga el modelo con `GET /garments/{garment_id}/model`. Antes se vuelve a comprobar que la prenda es del usuario.

```
outputs/garments/{garment_id}/
  model.glb        el modelo para la app
  thumbnail.png    la miniatura (foto sin fondo)
  photo.jpg        la foto usada
  metadata.json    estado, versión de la foto, fechas, tiempos y materiales aplicados
```

### Modo local (temporal)
Si `DC_API_URL` está vacía, el servicio arranca en **modo local**: no consulta al Digital Closet, la foto se envía en la misma petición (campo `image`) y la categoría se indica con `category`. El log lo muestra al iniciar: `Catálogo: MODO LOCAL`.

Para quitarlo cuando el Digital Closet esté desplegado:
1. Borrar `tryon/infrastructure/catalog/local_garment_catalog.py`.
2. En `tryon/bootstrap.py`, borrar la rama `else` de `build_catalog()`.
3. Opcional: quitar el campo `image` del endpoint `POST /garments/{garment_id}/model`.

El dominio y los casos de uso no cambian.

## Endpoints

Todas las rutas llevan el prefijo **`/api/v1/try-on`** (variable `API_PREFIX`). En la tabla se omite. `/health` responde también en la raíz.

| Método | Ruta | Qué hace |
|---|---|---|
| POST | `/garments/{garment_id}/model` | Genera (o regenera) el modelo de la prenda y su miniatura. Responde `201` con su estado. |
| GET | `/garments/{garment_id}/model` | Descarga el `.glb` (`model/gltf-binary`). Para la pantalla de detalle. |
| GET | `/garments/{garment_id}/model/info` | Estado: `status`, `badge`, `source_image_version`, `is_stale`, fechas, detalles. |
| GET | `/garments/{garment_id}/model/thumbnail` | Miniatura: la foto sin fondo, PNG transparente de 512 px. Para las tarjetas. |
| GET | `/garments/models?ids=a,b,c` | Estado, etiqueta y rutas de hasta 100 prendas en una llamada. Para la pantalla del armario. |
| GET | `/health` | `{"status": "ok"}` si está listo. Público y sin datos internos. |
| POST | `/generate` | Técnico: foto → `.glb` con todos los parámetros del motor, sin prenda asociada. |
| POST | `/check` | Técnico: valida una foto con Gemini (sin GPU). |
| GET | `/docs` | Documentación interactiva (Swagger). |

**Etiquetas de las tarjetas (`badge`):**

| `badge` | Cuándo | Texto sugerido en la app |
|---|---|---|
| `AR_READY` | Modelo listo y la categoría se puede probar en la cara | AR READY |
| `MODEL_3D` | Modelo listo, pero solo para ver en 3D | 3D |
| `null` | Sin modelo, o generándose | (sin etiqueta) |

Hoy solo los **lentes** son `AR_READY`; las gorras quedan como `MODEL_3D`. Se cambia con `AR_CATEGORIES` (por ejemplo `glasses,hat`), sin tocar código.

Flujo sugerido en la app: el armario llama a `GET /garments/models?ids=...`, muestra la miniatura de cada prenda con su etiqueta y, al abrir el detalle, descarga el `.glb` para el visor 3D y la prueba en RA.

Parámetros opcionales de `POST /garments/{garment_id}/model`:
- `category`: `glasses` o `hat`. Por defecto se deduce de la subcategoría. En modo local es obligatoria.
- `lens_type`: `clear`, `sun_gray`, `sun_brown`, `sun_green` o `mirror`. Por defecto se mide en la foto.
- `frame_material`: `glossy_plastic`, `matte_plastic` o `metal`.
- `image`: solo modo local.

Errores, con el mismo formato que el Digital Closet: `{"error": {"code", "message"}}`.

| Código HTTP | `code` |
|---|---|
| 400 | `INVALID_GARMENT_ID`, `INVALID_PHOTO` |
| 401 | `UNAUTHENTICATED` |
| 404 | `GARMENT_NOT_FOUND`, `MODEL_NOT_FOUND` |
| 409 | `GENERATION_IN_PROGRESS`, `MODEL_NOT_READY` |
| 422 | `UNSUPPORTED_ACCESSORY`, `GARMENT_WITHOUT_PHOTO` |
| 503 | `CATALOG_UNAVAILABLE` (el Digital Closet no responde) |
| 507 | `INSUFFICIENT_GPU_MEMORY` |

## Autenticación (token de IAM)

- **De dónde sale:** la app inicia sesión en IAM y recibe un token de acceso (JWT). Ese token va en cada llamada: `Authorization: Bearer <token>`.
- **Quién lo valida:** este servicio no lo valida por su cuenta. Lo reenvía al Digital Closet, que lo valida contra IAM (sesión activa) y responde `404` si la prenda no es del usuario. Si el token no sirve, este servicio responde `401 UNAUTHENTICATED`.
- **En Swagger:** botón **Authorize** (candado), se pega el token una vez y vale para todos los endpoints de `Garment Models`. Queda guardado aunque recargues la página.
- **Modo local:** no hace falta token.

## Seguridad: qué se expone

- **`/health` es mínimo:** solo `{"status": "ok"}` (o `starting` mientras carga el modelo). No muestra GPU, versiones, modelo ni configuración: esa información ayuda a un atacante a buscar vulnerabilidades conocidas (*information disclosure*).
- **Diagnóstico interno:** `GET /health/details` (oculto en Swagger) devuelve GPU, VRAM, modelo y catálogo. Solo existe si `DIAGNOSTICS_KEY` está configurada, y exige el header `X-Diagnostics-Key`. Si no, responde `404`.
- **Swagger:** en producción, `EXPOSE_DOCS=false` oculta `/docs`, `/redoc` y `/openapi.json`.

## API Gateway (futuro)

Hoy la app llama directo al servicio (`http://localhost:8000/api/v1/try-on/...`). Cuando exista el gateway:
- El gateway enruta `/api/v1/try-on/*` a este servicio (puerto 8000) y `/api/v1/garments/*` al Digital Closet. El prefijo propio evita choques.
- Debe reenviar el header `Authorization: Bearer …`: este servicio lo pasa al Digital Closet.
- Si el gateway publica el servicio con otra ruta (por ejemplo `/tryon`), basta con `PUBLIC_BASE_PATH=/tryon` para que `model_url` y `thumbnail_url` sean correctas. Sin cambios de código.
- La llamada de este servicio al Digital Closet (`DC_API_URL`) es interna: puede seguir directa, sin pasar por el gateway.
- Para el chequeo de salud del gateway: `GET /health`.

## Puesta en marcha

Requisitos: Docker Desktop con WSL2, GPU NVIDIA de 8 GB o más, y un token de Hugging Face con acceso aceptado a `stabilityai/stable-point-aware-3d`.

1. Copia `.env.example` como `.env` y completa `HF_TOKEN`. Para conectar el Digital Closet, completa `DC_API_URL`.
2. `docker compose up --build`. El primer build tarda de 15 a 30 minutos.
3. Espera `Modelo cargado` en el log.

Variables principales (`.env`):

| Variable | Valor |
|---|---|
| `HF_TOKEN` | Token de Hugging Face |
| `DC_API_URL` | `http://host.docker.internal:8081/api/v1` (Digital Closet en Windows). Vacía = modo local. |
| `RECON_MODEL` | `spar3d` (por defecto) o `sf3d` |
| `SPAR3D_LOW_VRAM` | `1` (necesario con 8 GB) |
| `AR_CATEGORIES` | Categorías con etiqueta `AR_READY`. Por defecto `glasses`. |
| `API_PREFIX` | Prefijo de las rutas. Por defecto `/api/v1/try-on`. |
| `DIAGNOSTICS_KEY` | Clave para `/health/details`. Vacía = desactivado. |
| `EXPOSE_DOCS` | `true` (por defecto) muestra Swagger; `false` en producción. |
| `PUBLIC_BASE_PATH` | Ruta con la que la app ve el servicio; se usa en `model_url` y `thumbnail_url`. Por defecto, igual a `API_PREFIX`. |
| `GEMINI_API_KEY` | Opcional: validación de fotos en `/generate` y `/check` |

## Probar desde PowerShell

Con el Digital Closet:
```
curl.exe -X POST "http://localhost:8000/api/v1/try-on/garments/<garment_id>/model" -H "Authorization: Bearer <token>"
curl.exe "http://localhost:8000/api/v1/try-on/garments/<garment_id>/model" -H "Authorization: Bearer <token>" -o lentes.glb
```

En modo local (el ID puede ser cualquier UUID):
```
curl.exe -X POST "http://localhost:8000/api/v1/try-on/garments/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90/model?category=glasses" -F "image=@lentes.jpg"
curl.exe "http://localhost:8000/api/v1/try-on/garments/3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90/model" -o lentes.glb
```

Consejo para las fotos de lentes: de frente, con la cámara a la altura de los lentes y las patillas abiertas.

## Pruebas

No usan la GPU: el motor 3D y el Digital Closet se reemplazan por dobles.
```
pip install pytest fastapi httpx python-multipart
python -m pytest tests
```

## Motor 3D (resumen)
- `recon.py`: SPAR3D o SF3D (`RECON_MODEL`), quitar fondo y reconstrucción. Remallado `quad` por defecto: superficie más lisa.
- `postprocess.py`, modo `light` (SPAR3D): orientar, separar y rehacer las lunas lisas, ajustar el color del marco a la foto, quitar piezas del fondo, normales suaves, tamaño real (lentes 14 cm, gorras 19.5 cm) y anclaje.
- `material_presets.py` + `materials/`: lunas claras, de sol (gris, marrón, verde) o espejo; marcos de plástico brillante, mate o metal (texturas de ambientCG, licencia CC0).
- `gemini_check.py`: validación opcional de la foto con Gemini.
