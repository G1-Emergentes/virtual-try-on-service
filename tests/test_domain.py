import pytest

from tryon.domain.errors import InvalidArFit, InvalidGarmentId, UnsupportedAccessory
from tryon.domain.garment_model import (AccessoryCategory, ArFit, ArFitSource, ArSupportPolicy,
                                       GarmentId, GarmentModel, ModelBadge, ModelStatus)

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


# --- Colocación en la cara (ar_fit) ------------------------------------------------------

def _ready(category=AccessoryCategory.GLASSES):
    model = GarmentModel.start(GarmentId(ID), category, 1)
    model.complete("j", {"seconds": 2})
    return model


def test_ar_fit_por_defecto_solo_en_modelos_ar_ready():
    policy = ArSupportPolicy()
    assert policy.fit_for(GarmentModel.start(GarmentId(ID), AccessoryCategory.GLASSES, 1)) is None
    assert policy.fit_for(_ready(AccessoryCategory.HAT)) is None
    fit = policy.fit_for(_ready())
    assert fit.anchor == "nose_bridge" and fit.source is ArFitSource.DEFAULT
    assert fit.offset_m == (0.0, 0.003, 0.006)


def test_ar_fit_por_defecto_configurable_y_validado():
    policy = ArSupportPolicy(default_fits={"glasses": {"offset_m": [0, 0.001, 0.004], "scale": 1.1}})
    assert policy.fit_for(_ready()).scale == 1.1
    with pytest.raises(InvalidArFit):                      # configuración absurda: no arranca
        ArSupportPolicy(default_fits={"glasses": {"offset_m": [0, 1, 0]}})


def test_calibrar_guarda_el_ajuste_en_el_modelo():
    policy = ArSupportPolicy()
    model = _ready()
    model.calibrate(policy.calibrated_fit(model, {"offset_m": [0, 0.004, 0.007], "pitch_deg": 2}))
    assert model.details["seconds"] == 2                   # no pisa los datos del motor
    fit = policy.fit_for(model)
    assert fit.source is ArFitSource.CALIBRATED and fit.pitch_deg == 2.0
    # un dato guardado corrupto no rompe la consulta: vuelve al de la categoría
    model.details["ar_fit"] = {"offset_m": "x"}
    assert policy.fit_for(model).source is ArFitSource.DEFAULT


def test_no_se_calibra_lo_que_no_es_ra_ni_lo_que_no_esta_listo():
    policy = ArSupportPolicy()
    with pytest.raises(UnsupportedAccessory):
        policy.calibrated_fit(_ready(AccessoryCategory.HAT), {"offset_m": [0, 0, 0]})
    model = GarmentModel.start(GarmentId(ID), AccessoryCategory.GLASSES, 1)
    with pytest.raises(ValueError):
        model.calibrate(ArFit("nose_bridge", (0, 0, 0)))


@pytest.mark.parametrize("data", [
    {"offset_m": [0, 0]}, {"offset_m": [0, 0, float("nan")]}, {"offset_m": [0, 0, 0], "scale": 0.1},
    {"offset_m": [0, 0, 0], "pitch_deg": 45}, {"offset_m": ["a", 0, 0]}, "no-es-un-objeto"])
def test_ar_fit_rechaza_valores_invalidos(data):
    with pytest.raises(InvalidArFit):
        ArFit.from_dict(data, "nose_bridge")
