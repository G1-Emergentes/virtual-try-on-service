import pytest

from tryon.application.generate_garment_model import GenerateGarmentModel, GenerateGarmentModelCommand
from tryon.application.get_garment_model import GetGarmentModel
from tryon.domain.errors import (GarmentNotFound, GenerationFailed, ModelNotFound, ModelNotReady,
                                 UnsupportedAccessory)
from tryon.domain.garment_model import AccessoryCategory, ModelStatus
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
