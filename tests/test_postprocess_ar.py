"""
Pruebas del motor (postprocess.py) para RA, con geometría sintética y sin GPU:
- lens_line_point: el origen de los lentes a la altura del centro de los cristales.
- hinge_depth / make_ar_variant: la variante sin patillas para el probador.

Los lentes de prueba miden como los reales (metros, ya normalizados): frente de 14 cm de
ancho y 1.2 cm de grosor en z = [-0.012, 0], patillas en los extremos hasta z = -0.14.
"""
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
postprocess = pytest.importorskip("postprocess")


def _box(extents, center):
    box = trimesh.creation.box(extents=extents,
                               transform=trimesh.transformations.translation_matrix(center))
    return box.subdivide_to_size(0.002)          # malla densa, como la de SPAR3D


def _frame(with_temples=True):
    parts = [_box([0.14, 0.04, 0.012], [0, 0, -0.006])]
    if with_temples:
        for x in (-0.0675, 0.0675):              # patillas: 5 mm de ancho en los extremos
            parts.append(_box([0.005, 0.005, 0.128], [x, 0.008, -0.076]))
    return trimesh.util.concatenate(parts)


def _lenses(y_center=-0.002):
    return trimesh.util.concatenate([_box([0.05, 0.035, 0.002], [x, y_center, -0.004])
                                     for x in (-0.034, 0.034)])


def _scene(frame, lenses):
    scene = trimesh.Scene()
    scene.add_geometry(frame, node_name="glasses", geom_name="glasses")
    scene.add_geometry(lenses, node_name="lenses", geom_name="lenses")
    return scene


def test_ancla_a_la_altura_del_centro_de_los_cristales():
    point, source = postprocess.lens_line_point(_frame(), _lenses(y_center=-0.003),
                                                postprocess.GlassesConfig())
    assert source == "lens_line"
    assert point[1] == pytest.approx(-0.003, abs=1e-4)      # y: centro de los cristales
    assert point[2] == pytest.approx(0.0, abs=1e-6)         # z: plano frontal
    assert abs(point[0]) < 1e-3                             # x: centro (puente)


def test_ancla_sin_cristales_usa_el_puente():
    _, source = postprocess.lens_line_point(_frame(), None, postprocess.GlassesConfig())
    assert source == "bridge"


def test_encuentra_la_bisagra_detras_del_frente():
    hinge = postprocess.hinge_depth(_frame())
    assert hinge is not None
    assert -0.018 < hinge < -0.011                          # justo detrás del frente (-0.012)


def test_variante_ar_corta_en_la_bisagra_y_conserva_cristales_y_origen():
    frame, lenses = _frame(), _lenses()
    scene = _scene(frame, lenses)
    ar, stats = postprocess.make_ar_variant(scene, postprocess.GlassesConfig())
    assert stats["method"] == "hinge"
    cut = ar.geometry["glasses"]
    assert cut.bounds[0, 2] > -0.025                        # sin patillas (antes llegaban a -0.14)
    assert cut.bounds[1, 2] == pytest.approx(0.0, abs=1e-6)                 # mismo frente
    assert cut.extents[0] == pytest.approx(0.14, abs=1e-3)                  # mismo ancho
    assert np.allclose(ar.geometry["lenses"].vertices, lenses.vertices)     # cristales intactos
    assert scene.geometry["glasses"].bounds[0, 2] == pytest.approx(-0.14)   # original intacto


def test_sin_bisagra_usa_crop_temples_y_lo_registra():
    ar, stats = postprocess.make_ar_variant(_scene(_frame(with_temples=False), _lenses()),
                                            postprocess.GlassesConfig())
    assert stats["method"] == "crop_temples"
    assert ar.geometry["glasses"].extents[2] == pytest.approx(0.012, abs=1e-3)
