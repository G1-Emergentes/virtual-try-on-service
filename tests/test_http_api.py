"""
Pruebas de la API HTTP de punta a punta, sin GPU:
- el Digital Closet se simula con httpx.MockTransport (mismas rutas y respuestas que el real);
- el motor 3D se reemplaza por FakeGenerator;
- el repositorio es el real, sobre una carpeta temporal.
"""
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tryon.application.generate_garment_model import GenerateGarmentModel
from tryon.application.get_garment_model import GetGarmentModel
from tryon.bootstrap import Container
from tryon.config import Settings
from tryon.infrastructure.catalog.digital_closet_catalog import DigitalClosetCatalog
from tryon.infrastructure.catalog.local_garment_catalog import LocalGarmentCatalog
from tryon.infrastructure.persistence.file_model_repository import FileModelRepository
from tryon.interfaces.http.errors import register_error_handlers
from tryon.interfaces.http.garment_model_router import router

from .fakes import GLASSES_ID, FakeGenerator

TOKEN = "user-token"
OTHER_USER_GARMENT = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
TOP_ID = "bbbbbbbb-bbbb-4ccc-8ddd-eeeeeeeeeeee"
DC = "http://dc.test/api/v1"
P = "/api/v1/try-on"      # API_PREFIX por defecto


def fake_digital_closet(request: httpx.Request) -> httpx.Response:
    """Simula el Digital Closet: GarmentReadModel, image-access y la URL firmada."""
    if request.url.host == "storage.test":
        return httpx.Response(200, content=b"\xff\xd8jpeg")
    if request.headers.get("Authorization") != f"Bearer {TOKEN}":
        return httpx.Response(401, json={"error": {"code": "UNAUTHENTICATED", "message": "token"}})
    path = request.url.path.removeprefix("/api/v1")
    garments = {
        GLASSES_ID: {"id": GLASSES_ID, "category": "ACCESSORY", "subcategory": "SUNGLASSES",
                     "image": {"id": "img", "version": 2}},
        TOP_ID: {"id": TOP_ID, "category": "TOP", "subcategory": None, "image": None},
    }
    if path == "/garments/resolve" and request.method == "POST":
        ids = json.loads(request.content)["ids"]
        results = []
        for i in ids:
            g = garments.get(i)
            results.append({"id": i, "status": "AVAILABLE" if g else "NOT_FOUND", "garment": g})
        return httpx.Response(200, json={"results": results})
    if path.startswith("/garments/"):
        parts = path.split("/")
        garment = garments.get(parts[2])
        if garment is None:
            return httpx.Response(404, json={"error": {"code": "GARMENT_NOT_FOUND", "message": "x"}})
        if len(parts) == 3:
            return httpx.Response(200, json=garment)
        if parts[3] == "image-access":
            return httpx.Response(200, json={"image_id": "img", "image_version": 2, "mime_type": "image/jpeg",
                                             "url": "https://storage.test/signed/photo.jpg"})
    return httpx.Response(500)


def make_client(tmp_path, catalog) -> TestClient:
    repo = FileModelRepository(str(tmp_path / "out"))
    generator = FakeGenerator(tmp_path)
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(router, prefix=P)
    settings = Settings(str(tmp_path / "out"), 1024, 10, "" if isinstance(catalog, LocalGarmentCatalog) else DC, 5)
    app.state.container = Container(settings=settings, pipeline=None,
                                    generate_garment_model=GenerateGarmentModel(catalog, repo, generator),
                                    get_garment_model=GetGarmentModel(catalog, repo))
    return TestClient(app)


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path, DigitalClosetCatalog(DC, transport=httpx.MockTransport(fake_digital_closet)))


AUTH = {"Authorization": f"Bearer {TOKEN}"}


def test_flujo_completo_con_digital_closet(client, tmp_path):
    r = client.post(f"{P}/garments/{GLASSES_ID}/model", headers=AUTH)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["garment_id"] == GLASSES_ID and body["status"] == "READY"
    assert body["category"] == "glasses" and body["source_image_version"] == 2
    assert body["model_url"] == f"{P}/garments/{GLASSES_ID}/model"

    # el modelo, la foto y la metadata quedan en la carpeta de la prenda
    folder = tmp_path / "out" / "garments" / GLASSES_ID
    assert (folder / "model.glb").read_bytes() == b"glTF-fake"
    assert (folder / "photo.jpg").exists()
    assert json.loads((folder / "metadata.json").read_text())["status"] == "READY"

    r = client.get(f"{P}/garments/{GLASSES_ID}/model", headers=AUTH)
    assert r.status_code == 200 and r.content == b"glTF-fake"
    assert r.headers["content-type"] == "model/gltf-binary"
    assert r.headers["x-garment-id"] == GLASSES_ID and r.headers["x-model-stale"] == "no"

    r = client.get(f"{P}/garments/{GLASSES_ID}/model/info", headers=AUTH)
    assert r.status_code == 200 and r.json()["status"] == "READY"
    assert r.json()["badge"] == "AR_READY"
    assert r.json()["thumbnail_url"] == f"{P}/garments/{GLASSES_ID}/model/thumbnail"

    r = client.get(f"{P}/garments/{GLASSES_ID}/model/thumbnail", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert (folder / "thumbnail.png").exists()


def test_consulta_en_lote_para_el_armario(client):
    client.post(f"{P}/garments/{GLASSES_ID}/model", headers=AUTH)
    r = client.get(f"{P}/garments/models?ids={GLASSES_ID},{TOP_ID},{OTHER_USER_GARMENT}", headers=AUTH)
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["garment_id"] for i in items] == [GLASSES_ID, TOP_ID, OTHER_USER_GARMENT]
    assert items[0]["badge"] == "AR_READY" and items[0]["thumbnail_url"]
    assert items[1]["found"] is True and items[1]["badge"] is None       # sin modelo
    assert items[2]["found"] is False                                    # no es del usuario


def test_gorra_es_solo_3d(tmp_path):
    c = make_client(tmp_path, LocalGarmentCatalog())
    r = c.post(f"{P}/garments/{GLASSES_ID}/model?category=hat", files={"image": ("g.jpg", b"x", "image/jpeg")})
    assert r.status_code == 201 and r.json()["badge"] == "MODEL_3D"


@pytest.mark.parametrize("garment_id,headers,status,code", [
    (GLASSES_ID, {}, 401, "UNAUTHENTICATED"),
    (GLASSES_ID, {"Authorization": "Bearer otro"}, 401, "UNAUTHENTICATED"),
    (OTHER_USER_GARMENT, AUTH, 404, "GARMENT_NOT_FOUND"),
    (TOP_ID, AUTH, 422, "UNSUPPORTED_ACCESSORY"),
    ("no-es-uuid", AUTH, 400, "INVALID_GARMENT_ID"),
])
def test_errores_de_generacion(client, garment_id, headers, status, code):
    r = client.post(f"{P}/garments/{garment_id}/model", headers=headers)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code


def test_descargar_sin_generar(client):
    r = client.get(f"{P}/garments/{GLASSES_ID}/model", headers=AUTH)
    assert r.status_code == 404 and r.json()["error"]["code"] == "MODEL_NOT_FOUND"


def test_con_digital_closet_no_se_acepta_foto_suelta(client):
    r = client.post(f"{P}/garments/{GLASSES_ID}/model", headers=AUTH,
                    files={"image": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_PHOTO"


def test_digital_closet_caido(tmp_path):
    def down(request):
        raise httpx.ConnectError("sin conexión")
    c = make_client(tmp_path, DigitalClosetCatalog(DC, transport=httpx.MockTransport(down)))
    r = c.post(f"{P}/garments/{GLASSES_ID}/model", headers=AUTH)
    assert r.status_code == 503 and r.json()["error"]["code"] == "CATALOG_UNAVAILABLE"


# --- Modo local ----------------------------------------------------------------------------

def test_modo_local(tmp_path):
    c = make_client(tmp_path, LocalGarmentCatalog())
    r = c.post(f"{P}/garments/{GLASSES_ID}/model?category=glasses",
               files={"image": ("lentes.jpg", b"\xff\xd8jpeg", "image/jpeg")})
    assert r.status_code == 201, r.text
    assert r.json()["source_image_version"] is None
    assert c.get(f"{P}/garments/{GLASSES_ID}/model").content == b"glTF-fake"


def test_modo_local_exige_foto_y_categoria(tmp_path):
    c = make_client(tmp_path, LocalGarmentCatalog())
    r = c.post(f"{P}/garments/{GLASSES_ID}/model", files={"image": ("l.jpg", b"x", "image/jpeg")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "UNSUPPORTED_ACCESSORY"
    r = c.post(f"{P}/garments/{GLASSES_ID}/model?category=hat")
    assert r.status_code == 422 and r.json()["error"]["code"] == "GARMENT_WITHOUT_PHOTO"


def test_urls_con_ruta_publica_del_gateway(tmp_path):
    """Si el gateway publica el servicio en otra ruta, las URLs devueltas la usan."""
    c = make_client(tmp_path, LocalGarmentCatalog())
    c.app.state.container = c.app.state.container.__class__(
        **{**c.app.state.container.__dict__,
           "settings": Settings(str(tmp_path / "out"), 1024, 10, "", 5, public_base_path="/tryon")})
    r = c.post(f"{P}/garments/{GLASSES_ID}/model?category=glasses", files={"image": ("l.jpg", b"x", "image/jpeg")})
    assert r.json()["model_url"] == f"/tryon/garments/{GLASSES_ID}/model"


# --- Health: no expone datos internos -------------------------------------------------------

def _health_client(tmp_path, key=""):
    from tryon.interfaces.http.generation_router import health_router
    app = FastAPI()
    app.include_router(health_router)
    pipeline = type("P", (), {"device": "cuda", "model_name": "spar3d"})()
    app.state.container = Container(settings=Settings(str(tmp_path), 1024, 10, "", 5, diagnostics_key=key),
                                    pipeline=pipeline, generate_garment_model=None, get_garment_model=None)
    return TestClient(app)


def test_health_publico_es_minimo(tmp_path):
    assert _health_client(tmp_path).get("/health").json() == {"status": "ok"}


def test_diagnostico_oculto_sin_clave(tmp_path):
    assert _health_client(tmp_path).get("/health/details").status_code == 404
    c = _health_client(tmp_path, key="secreta")
    assert c.get("/health/details", headers={"X-Diagnostics-Key": "otra"}).status_code == 404
    r = c.get("/health/details", headers={"X-Diagnostics-Key": "secreta"})
    assert r.status_code == 200 and r.json()["model"] == "spar3d"
