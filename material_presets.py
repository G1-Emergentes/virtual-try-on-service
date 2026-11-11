"""
Biblioteca de materiales (materials/materials.json + materials/textures/).

- Lunas: presets de color, opacidad y brillo (clara, sol gris/marrón/verde, espejo).
- Marcos: el color sale de la foto; la textura (ambientCG, CC0) aporta rugosidad y
  relieve fino. Se "hornea" con proyección triplanar en el atlas UV del modelo:
  la textura se proyecta desde X, Y y Z según la posición 3D de cada punto, así
  queda continua aunque las UV de SPAR3D vengan en pedazos.
"""
import json
import os
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy import ndimage

MATERIALS_DIR = os.getenv("MATERIALS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "materials"))


@lru_cache(maxsize=1)
def load_library():
    with open(os.path.join(MATERIALS_DIR, "materials.json"), encoding="utf-8") as f:
        return json.load(f)


def lens_preset(lens_type):
    lib = load_library()["lenses"]
    return lib.get(lens_type) if lens_type else None


def frame_preset(frame_material, category):
    lib = load_library()
    name = frame_material if frame_material in lib["frames"] else lib["defaults"].get(category)
    return name, lib["frames"].get(name)


@lru_cache(maxsize=8)
def _load_maps(prefix):
    """Mapas de la textura como arreglos float 0-1 (los que existan)."""
    maps = {}
    for key, suffix in (("roughness", "_Roughness.jpg"), ("normal", "_NormalGL.jpg"),
                        ("metalness", "_Metalness.jpg"), ("color", "_Color.jpg")):
        path = os.path.join(MATERIALS_DIR, "textures", prefix + suffix)
        if os.path.exists(path):
            img = Image.open(path).convert("L" if key in ("roughness", "metalness") else "RGB")
            img = img.resize((512, 512), Image.BILINEAR)          # suficiente para un celular
            maps[key] = np.asarray(img).astype(np.float32) / 255
    return maps


def _sample(img, u, v):
    h, w = img.shape[:2]
    x = ((u % 1.0) * (w - 1)).astype(int)
    y = ((v % 1.0) * (h - 1)).astype(int)
    return img[y, x]


def _texel_geometry(mesh, size):
    """Para cada texel del atlas cubierto por una cara: posición 3D, normal y marco tangente."""
    from projection import _raster_triangle

    uvpx = np.c_[mesh.visual.uv[:, 0] * size, (1 - mesh.visual.uv[:, 1]) * size]
    v = mesh.vertices
    uv = np.asarray(mesh.visual.uv)
    fn = mesh.face_normals
    xs_all, ys_all, pos_all, frame_all = [], [], [], []
    for fi, f in enumerate(mesh.faces):
        r = _raster_triangle(uvpx[f], size, size)
        if r is None:
            continue
        xs, ys, bary = r
        p = bary @ v[f]
        # tangente y bitangente a partir de las UV (cómo se orienta el mapa de normales)
        e1, e2 = v[f[1]] - v[f[0]], v[f[2]] - v[f[0]]
        d1, d2 = uv[f[1]] - uv[f[0]], uv[f[2]] - uv[f[0]]
        det = d1[0] * d2[1] - d2[0] * d1[1]
        if abs(det) < 1e-12:
            continue
        t = (e1 * d2[1] - e2 * d1[1]) / det
        b = (e2 * d1[0] - e1 * d2[0]) / det
        n = fn[fi]
        t = t - n * (t @ n)
        t /= max(np.linalg.norm(t), 1e-12)
        b = np.cross(n, t) * np.sign(np.cross(n, t) @ b or 1.0)
        xs_all.append(xs)
        ys_all.append(ys)
        pos_all.append(p)
        frame_all.append(np.repeat(np.stack([t, b, n])[None], len(xs), 0))
    if not xs_all:
        return None
    return (np.concatenate(xs_all), np.concatenate(ys_all),
            np.concatenate(pos_all), np.concatenate(frame_all))


def bake_frame_material(mesh, preset, size=1024):
    """
    Devuelve (metallicRoughness, normal, base_color) como imágenes del tamaño del atlas.
    base_color solo cambia en metales (detalle de cepillado sobre el color de la foto).
    """
    maps = _load_maps(preset["texture"])
    if not maps:
        return None
    geo = _texel_geometry(mesh, size)
    if geo is None:
        return None
    xs, ys, pos, frame = geo
    nrm = frame[:, 2]
    # pesos triplanares: cada eje cuenta según cuánto mira la superficie hacia él
    w = np.abs(nrm) ** 4
    w /= w.sum(1, keepdims=True)
    q = pos / preset.get("tile_m", 0.04)
    planes = ((q[:, 1], q[:, 2]), (q[:, 0], q[:, 2]), (q[:, 0], q[:, 1]))   # proyección X, Y, Z

    def tri(img):
        out = 0
        for k, (a, b) in enumerate(planes):
            s = _sample(img, a, b)
            out = out + (w[:, k:k + 1] if s.ndim > 1 else w[:, k]) * s
        return out

    mr = np.zeros((size, size, 3), np.float32)
    mr[..., 1] = 0.5
    filled = np.zeros((size, size), bool)
    rough = tri(maps["roughness"]) if "roughness" in maps else np.full(len(xs), 0.5)
    metal = tri(maps["metalness"]) if "metalness" in maps else np.zeros(len(xs))
    mr[ys, xs, 1] = np.clip(rough * preset.get("roughness_scale", 1.0), 0.03, 1)
    mr[ys, xs, 2] = np.clip(metal * preset.get("metallic", 0.0), 0, 1) if "metalness" in maps \
        else preset.get("metallic", 0.0)
    filled[ys, xs] = True

    normal = None
    if "normal" in maps:
        strength = preset.get("normal_strength", 0.3)
        # perturbación en el mundo: mezcla de las 3 proyecciones (método "UDN")
        world = nrm.copy()
        axes = ((1, 2), (0, 2), (0, 1))
        for k, (i, j) in enumerate(axes):
            a, b = planes[k]
            tn = _sample(maps["normal"], a, b) * 2 - 1
            world[:, i] += w[:, k] * tn[:, 0] * strength
            world[:, j] += w[:, k] * tn[:, 1] * strength
        world /= np.maximum(np.linalg.norm(world, axis=1, keepdims=True), 1e-9)
        # al espacio tangente de cada cara (lo que espera glTF)
        tan = np.einsum("nij,nj->ni", frame, world)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        normal = np.zeros((size, size, 3), np.float32)
        normal[..., 2] = 1.0
        normal[ys, xs] = tan
        normal = normal * 0.5 + 0.5

    detail = None
    if "color" in maps and preset.get("color_detail", 0) > 0:
        c = tri(maps["color"])
        lum = c @ np.array([0.299, 0.587, 0.114], np.float32)
        detail = np.ones((size, size), np.float32)
        detail[ys, xs] = 1 + preset["color_detail"] * (lum / max(lum.mean(), 1e-3) - 1)

    # extender los bordes de cada isla UV (evita líneas por filtrado)
    idx = ndimage.distance_transform_edt(~filled, return_distances=False, return_indices=True)
    near = ndimage.binary_dilation(filled, iterations=4)
    mr = np.where(near[..., None], mr[idx[0], idx[1]], mr)
    if normal is not None:
        normal = np.where(near[..., None], normal[idx[0], idx[1]], normal)
    if detail is not None:
        detail = np.where(near, detail[idx[0], idx[1]], detail)

    to_img = lambda a: Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return to_img(mr), (to_img(normal) if normal is not None else None), detail
