import pytest

from tryon.application.generate_garment_model import GenerateGarmentModel, GenerateGarmentModelCommand
from tryon.application.get_garment_model import GetGarmentModel
from tryon.domain.errors import (GarmentNotFound, GenerationFailed, ModelNotFound, ModelNotReady,
                                 UnsupportedAccessory)
from tryon.domain.garment_model import AccessoryCategory, ModelStatus, ModelVariant
from tryon.domain.ports import Credentials

from .fakes import GLASSES_ID, FakeCatalog, FakeGenerator, InMemoryRepository

BAG_ID = "11111111-2222-4333-8444-555555555555"
CREDS = Credentials("token")


@pytest.fixture
def catalog():
    return FakeCatalog({GLASSES_ID: ("SUNGLASSES", 3), BAG_ID: ("HANDBAG", 1)})


def test_genera_y_asocia_el_modelo_a_la_prenda(catalog, tmp_path):
    repo, gen = InMemoryRepository(), FakeGenerator(tmp_path)
    model = GenerateGarmentModel(catalog, repo, gen).execute(GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    assert model.status is ModelStatus.READY
    assert model.category is AccessoryCategory.GLASSES        # deducida de SUNGLASSES
    assert model.source_image_version == 3
    assert repo.history == ["PROCESSING", "READY"]
    assert GLASSES_ID in repo.files and GLASSES_ID in repo.photos


def test_la_categoria_indicada_manda(catalog, tmp_path):
    gen = FakeGenerator(tmp_path)
    cmd = GenerateGarmentModelCommand(BAG_ID, CREDS, category=AccessoryCategory.HAT)
    GenerateGarmentModel(catalog, InMemoryRepository(), gen).execute(cmd)
    assert gen.calls[0][0] is AccessoryCategory.HAT


def test_rechaza_accesorios_no_soportados(catalog, tmp_path):
    with pytest.raises(UnsupportedAccessory):
        GenerateGarmentModel(catalog, InMemoryRepository(), FakeGenerator(tmp_path)).execute(
            GenerateGarmentModelCommand(BAG_ID, CREDS))


def test_prenda_inexistente(catalog, tmp_path):
    with pytest.raises(GarmentNotFound):
        GenerateGarmentModel(catalog, InMemoryRepository(), FakeGenerator(tmp_path)).execute(
            GenerateGarmentModelCommand("99999999-2222-4333-8444-555555555555", CREDS))


def test_si_falla_queda_registrado(catalog, tmp_path):
    repo = InMemoryRepository()
    with pytest.raises(GenerationFailed):
        GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path, fail=True)).execute(
            GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    assert repo.history == ["PROCESSING", "FAILED"]
    with pytest.raises(ModelNotReady):
        GetGarmentModel(catalog, repo).file(GLASSES_ID, CREDS)


def test_consulta_y_detecta_foto_nueva(catalog, tmp_path):
    repo = InMemoryRepository()
    GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path)).execute(GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    getter = GetGarmentModel(catalog, repo)
    assert getter.file(GLASSES_ID, CREDS).is_stale is False
    catalog.garments[GLASSES_ID] = ("SUNGLASSES", 4)          # subieron otra foto
    assert getter.info(GLASSES_ID, CREDS).is_stale is True


def test_sin_modelo(catalog):
    with pytest.raises(ModelNotFound):
        GetGarmentModel(catalog, InMemoryRepository()).info(GLASSES_ID, CREDS)


def test_consulta_en_lote(catalog, tmp_path):
    repo = InMemoryRepository()
    GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path)).execute(GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    missing = "99999999-2222-4333-8444-555555555555"
    rows = GetGarmentModel(catalog, repo).many([GLASSES_ID, BAG_ID, missing], CREDS)
    assert [r.garment_id for r in rows] == [GLASSES_ID, BAG_ID, missing]     # mismo orden
    assert rows[0].badge.value == "AR_READY" and rows[0].has_thumbnail
    assert rows[1].found and rows[1].model is None and rows[1].badge is None
    assert not rows[2].found


# --- Variante para RA (sin patillas) ----------------------------------------------------

def test_guarda_la_variante_ar_y_anota_como_se_corto(catalog, tmp_path):
    repo = InMemoryRepository()
    model = GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path, ar_method="hinge")).execute(
        GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    assert GLASSES_ID in repo.files and GLASSES_ID in repo.ar_files
    assert model.details["ar_variant"] == {"method": "hinge", "depth_m": 0.02}
    assert model.details["seconds"] == 1.0                    # los datos del motor se conservan


def test_registra_el_respaldo_crop_temples(catalog, tmp_path):
    repo = InMemoryRepository()
    model = GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path, ar_method="crop_temples")).execute(
        GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    assert model.details["ar_variant"]["method"] == "crop_temples"


def test_descarga_por_variante(catalog, tmp_path):
    repo = InMemoryRepository()
    GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path, ar_method="hinge")).execute(
        GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    getter = GetGarmentModel(catalog, repo)
    full = getter.file(GLASSES_ID, CREDS)
    assert full.variant is ModelVariant.FULL and full.file_path.name == "job.glb"
    ar = getter.file(GLASSES_ID, CREDS, ModelVariant.AR)
    assert ar.variant is ModelVariant.AR and ar.file_path.name == "job_ar.glb"
    assert ar.has_ar_variant


def test_modelo_antiguo_sin_variante_ar_entrega_el_completo(catalog, tmp_path):
    repo = InMemoryRepository()
    GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path)).execute(
        GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    view = GetGarmentModel(catalog, repo).file(GLASSES_ID, CREDS, ModelVariant.AR)
    assert view.variant is ModelVariant.FULL and view.file_path.name == "job.glb"
    assert not view.has_ar_variant


def test_archivo_ar_perdido_entrega_el_completo(catalog, tmp_path):
    repo = InMemoryRepository()
    GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path, ar_method="hinge")).execute(
        GenerateGarmentModelCommand(GLASSES_ID, CREDS))
    repo.ar_files.clear()
    assert GetGarmentModel(catalog, repo).file(GLASSES_ID, CREDS, ModelVariant.AR).variant is ModelVariant.FULL


def test_variante_ar_solo_para_modelos_ar_ready(catalog, tmp_path):
    repo = InMemoryRepository()
    GenerateGarmentModel(catalog, repo, FakeGenerator(tmp_path, ar_method="hinge")).execute(
        GenerateGarmentModelCommand(BAG_ID, CREDS, category=AccessoryCategory.HAT))   # gorra: MODEL_3D
    getter = GetGarmentModel(catalog, repo)
    assert getter.file(BAG_ID, CREDS).variant is ModelVariant.FULL
    with pytest.raises(UnsupportedAccessory):
        getter.file(BAG_ID, CREDS, ModelVariant.AR)
