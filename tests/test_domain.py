import pytest

from tryon.domain.errors import InvalidGarmentId
from tryon.domain.garment_model import (AccessoryCategory, ArSupportPolicy, GarmentId, GarmentModel,
                                       ModelBadge, ModelStatus)

ID = "3f2a9c1e-5b7d-4e8a-9c21-7d4e5f6a8b90"


def test_garment_id_acepta_uuid_y_lo_normaliza():
    assert GarmentId(ID.upper()).value == ID


@pytest.mark.parametrize("bad", ["", "123", "../etc/passwd", "3f2a9c1e5b7d4e8a9c217d4e5f6a8b90x"])
def test_garment_id_rechaza_lo_que_no_es_uuid(bad):
    with pytest.raises(InvalidGarmentId):
        GarmentId(bad)


@pytest.mark.parametrize("sub,expected", [
    ("SUNGLASSES", AccessoryCategory.GLASSES), ("glasses", AccessoryCategory.GLASSES),
    ("CAP", AccessoryCategory.HAT), ("HANDBAG", None), (None, None)])
def test_subcategoria_a_categoria(sub, expected):
    assert AccessoryCategory.from_subcategory(sub) is expected


def test_ciclo_de_vida_del_modelo():
    model = GarmentModel.start(GarmentId(ID), AccessoryCategory.GLASSES, source_image_version=2)
    assert model.status is ModelStatus.PROCESSING
    model.complete("job1", {"seconds": 3})
    assert model.is_ready and model.job_id == "job1"
    with pytest.raises(ValueError):          # READY no puede volver a fallar
        model.fail("x")


def test_modelo_desactualizado_si_cambia_la_foto():
    model = GarmentModel.start(GarmentId(ID), AccessoryCategory.HAT, source_image_version=1)
    assert model.is_stale(2)
    assert not model.is_stale(1)
    assert not model.is_stale(None)           # versión desconocida (modo local)


def test_etiqueta_segun_politica_de_ra():
    policy = ArSupportPolicy(frozenset({"glasses"}))
    lentes = GarmentModel.start(GarmentId(ID), AccessoryCategory.GLASSES, 1)
    gorra = GarmentModel.start(GarmentId(ID), AccessoryCategory.HAT, 1)
    assert lentes.badge(policy) is None                      # aún no está listo
    lentes.complete("j", {})
    gorra.complete("j", {})
    assert lentes.badge(policy) is ModelBadge.AR_READY
    assert gorra.badge(policy) is ModelBadge.MODEL_3D        # solo 3D, sin RA por ahora
