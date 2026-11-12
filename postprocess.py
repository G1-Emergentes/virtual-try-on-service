"""
Posprocesado de accesorios generados por SF3D para Realidad Aumentada.

Pasos comunes a todas las categorías:
  1. Orientar: el frente del objeto mira hacia +Z (convención glTF).
  2. Colores: ajustar la textura a los colores de la foto (todas menos lentes) y
     armonizar: SF3D inventa gris claro en las zonas que la cámara
     no vio (cantos, costados, interior). Esas zonas heredan la paleta del
     frente (histogram matching) y se atenúan los brillos horneados.
     (Opcional, experimental) Proyectar la foto original sobre la textura.
  3. Suavizar la malla (filtro Taubin) sin abrir costuras de UV.
  4. Material según categoría. SF3D suele exportar metallic=1, roughness=0
     (un espejo) y un mapa de normales ruidoso: se reemplazan.
  5. Normalizar: tamaño real en metros y origen en el punto de anclaje.
  6. Limitar texturas a 1024 px para móvil.

Pasos solo para lentes:
  - Detectar cristales y separarlos en una malla transparente.
  - Recortar patillas en modo full (en light, SPAR3D las genera bien y se conservan).
  - Variante para RA (make_ar_variant): el mismo modelo cortado en las bisagras, para el
    probador. El visor 3D usa el modelo completo.

Categorías y anclaje (el origen 0,0,0 del .glb es el punto de anclaje):
  glasses  -> puente, a la altura    ancho  14 cm
              del centro de los
              cristales (plano frontal)
  hat      -> centro de la base     ancho  21 cm
  earrings -> parte superior        alto  4.5 cm
  generic  -> centro de la base     lado mayor 30 cm (visor 3D / piso)
"""
from dataclasses import dataclass, field

from typing import Optional

import numpy as np
import scipy.sparse as sp
import trimesh
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from PIL import Image
from trimesh.visual.material import PBRMaterial


# ---------- configuración ----------
@dataclass
class CategoryConfig:
    name: str
    size_m: float               # tamaño real objetivo
    size_axis: str              # "x" = ancho, "y" = alto, "max" = lado mayor
    anchor: str                 # "bridge", "base", "top"
    metallic: float
    roughness: float
    smooth_iterations: int
    min_depth_ratio: float = 0.0  # profundidad mínima / ancho (corrige modelos aplastados)
    match_photo: bool = True      # ajustar colores a la foto


CATEGORIES = {
    "glasses": CategoryConfig("glasses", 0.14, "x", "bridge", 0.0, 0.45, 8),
    "hat": CategoryConfig("hat", 0.195, "x", "base", 0.0, 0.85, 6, min_depth_ratio=0.80),
    "earrings": CategoryConfig("earrings", 0.045, "y", "top", 0.9, 0.30, 3),
    "generic": CategoryConfig("generic", 0.30, "max", "base", 0.0, 0.60, 4),
}


@dataclass
class GlassesConfig:
    front_depth_ratio: float = 0.10   # grosor del frente, relativo al ancho
    lens_seed_lum: float = 0.80       # luminancia de semilla del cristal
    lens_grow_lum: float = 0.45       # luminancia mínima para expandir
    frame_max_lum: float = 0.30       # más oscuro que esto = aro del marco
    hull_shrink: float = 0.96         # contrae el contorno del cristal
    keep_temples: bool = False
    temple_keep_ratio: float = 0.12   # cuánto de patilla conservar (bisagra)
    lens_alpha: float = 0.15
    lens_inset_ratio: float = 0.03    # cuánto se mete el cristal detrás del aro (relativo al ancho)
    lens_depth_ratio: float = 0.22    # profundidad máxima de un cristal curvo (más atrás: patillas)


@dataclass
class PostprocessOptions:
    category: str = "glasses"
    keep_normal_map: bool = False     # el de SF3D suele ser ruido
    harmonize_colors: bool = True     # zonas no vistas heredan el color del frente
    seen_facing: float = 0.65         # normal.z mínima para considerar "vista"
    highlight_keep: float = 0.25      # cuánto conservar de los brillos horneados
    highlight_pct: float = 50         # percentil del frente desde el que se considera brillo
    mode: str = "full"                # "full" = corrige defectos de SF3D; "light" = respeta la malla (SPAR3D)
    lens_type: Optional[str] = None   # preset de luna (Gemini): clear, sun_gray, sun_brown, sun_green, mirror
    frame_material: Optional[str] = None  # preset de marco (Gemini): glossy_plastic, matte_plastic, metal
    use_material_library: bool = True
    light_smooth_factor: int = 3      # modo light: suavizado extra (quita el aspecto "grumoso")
    light_subdiv_smooth: int = 10
    normal_diffuse: int = 6           # modo light: normales más suaves (reflejos sin "grumos")
    frame_chroma_keep: float = 0.35   # lentes: cuánto conservar de las manchas de color del marco
    frame_lum_keep: float = 0.7
    clear_frame_alpha: float = 0.55   # marcos transparentes
    project_photo: bool = True        # lentes: foto real sobre el frente del marco
    unlit: bool = False               # material sin iluminación: colores idénticos a la textura
    max_texture_px: int = 1024
    max_faces: int = 90000            # límite para subdividir (rendimiento en celular)
    glasses: GlassesConfig = field(default_factory=GlassesConfig)


# ---------- utilidades ----------
def _face_colors(mesh):
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(float) / 255
    uv = mesh.visual.uv[mesh.faces].mean(1)
    h, w = tex.shape[:2]
    x = np.clip((uv[:, 0] * (w - 1)).astype(int), 0, w - 1)
    y = np.clip(((1 - uv[:, 1]) * (h - 1)).astype(int), 0, h - 1)
    return tex[y, x]


def _limit(img, max_px):
    if img is None or max(img.size) <= max_px:
        return img
    return img.resize((max_px, max_px))


def _welded_index(mesh):
    """Índice de vértices soldados por posición (SF3D duplica vértices en las costuras UV)."""
    _, inv = np.unique(np.round(mesh.vertices, 5), axis=0, return_inverse=True)
    return inv.ravel()


def _face_adjacency(mesh, inv):
    welded = trimesh.Trimesh(np.zeros((inv.max() + 1, 3)), inv[mesh.faces], process=False)
    return welded.face_adjacency


def _submesh(mesh, mask):
    return mesh.submesh([np.where(mask)[0]], append=True)


def _to_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        geoms = list(scene_or_mesh.geometry.values())
        return geoms[0] if len(geoms) == 1 else scene_or_mesh.dump(concatenate=True)
    return scene_or_mesh


# ---------- pasos comunes ----------
def orient_front_to_positive_z(mesh, category="glasses"):
    """
    SF3D deja el lado que vio la cámara hacia -Z (observado en pruebas).
    En lentes se verifica con el puente nasal; en el resto se aplica la convención.
    """
    flip = True
    if category == "glasses":
        c = mesh.triangles_center
        width = mesh.extents[0]
        bridge = np.abs(c[:, 0] - mesh.bounds.mean(0)[0]) < 0.05 * width
        zmin, zmax = mesh.bounds[:, 2]
        flip = np.median(c[bridge, 2]) < (zmin + zmax) / 2
    if flip:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    return mesh


def taubin_smooth(mesh, iterations, lamb=0.5, mu=-0.53):
    if iterations <= 0:
        return mesh
    inv = _welded_index(mesh)
    n = inv.max() + 1
    pos = np.zeros((n, 3))
    pos[inv] = mesh.vertices
    edges = inv[mesh.edges_unique]
    edges = edges[edges[:, 0] != edges[:, 1]]
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    W = sp.diags(1 / deg) @ A
    for _ in range(iterations):
        pos = pos + lamb * (W @ pos - pos)
        pos = pos + mu * (W @ pos - pos)
    mesh.vertices = pos[inv]
    return mesh


def subdivide_textured(mesh, max_faces=60000, smooth_iterations=3, levels=1):
    """
    Divide cada triángulo en 4 (conservando UV) y suaviza: siluetas más curvas.
    Solo si el resultado no supera max_faces (el celular debe renderizarlo fluido).
    """
    if len(mesh.faces) * 4 ** levels > max_faces:
        if levels > 1:
            return subdivide_textured(mesh, max_faces, smooth_iterations, levels - 1)
        return mesh, False
    for _ in range(levels - 1):
        mesh, _ = subdivide_textured(mesh, max_faces=10 ** 9, smooth_iterations=0)
    material = mesh.visual.material
    v, uv, f = mesh.vertices, np.asarray(mesh.visual.uv), mesh.faces
    # subdivisión por punto medio propia: la de trimesh 4.4.1 interpola mal las UV
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    inv = inv.ravel()
    n_f = len(f)
    mid = len(v) + inv
    ab, bc, ca = mid[:n_f], mid[n_f:2 * n_f], mid[2 * n_f:]
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    new_f = np.vstack([np.c_[a, ab, ca], np.c_[ab, b, bc], np.c_[ca, bc, c], np.c_[ab, bc, ca]])
    new_v = np.vstack([v, (v[uniq[:, 0]] + v[uniq[:, 1]]) / 2])
    new_uv = np.vstack([uv, (uv[uniq[:, 0]] + uv[uniq[:, 1]]) / 2])
    out = trimesh.Trimesh(new_v, new_f, process=False)
    out.visual = trimesh.visual.TextureVisuals(uv=new_uv, material=material)
    return taubin_smooth(out, smooth_iterations), True


def smooth_normals(mesh, diffuse=0):
    """
    Normales por vértice promediando las caras vecinas (ponderadas por área).
    Se sueldan los vértices duplicados en las costuras UV para que no se vean líneas.
    Sin normales, el visor ilumina cada triángulo plano y el modelo se ve poligonal.
    """
    inv = _welded_index(mesh)
    tri = mesh.vertices[mesh.faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # largo = 2 x área
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, inv[mesh.faces[:, k]], fn)
    acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    if diffuse > 0:
        # difundir las normales entre vecinos: los reflejos se ven lisos aunque la
        # superficie tenga ondulaciones pequeñas (la silueta no cambia)
        edges = inv[mesh.edges_unique]
        edges = edges[edges[:, 0] != edges[:, 1]]
        n = len(acc)
        A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                          shape=(n, n)).tocsr()
        for _ in range(diffuse):
            acc = acc + A @ acc
            acc /= np.maximum(np.linalg.norm(acc, axis=1, keepdims=True), 1e-12)
    mesh.vertex_normals = acc[inv]
    return mesh


def apply_material(mesh, src_mat, cat, opts):
    mesh.visual.material = PBRMaterial(
        name=cat.name,
        baseColorTexture=_limit(src_mat.baseColorTexture, opts.max_texture_px),
        normalTexture=_limit(getattr(src_mat, "normalTexture", None), opts.max_texture_px)
        if opts.keep_normal_map else None,
        metallicFactor=cat.metallic,
        roughnessFactor=cat.roughness,
        doubleSided=True,
    )


def normalize(meshes, cat, anchor_point=None):
    """Escala a tamaño real y mueve el punto de anclaje al origen."""
    ref = meshes[0]
    ext = ref.extents
    current = {"x": ext[0], "y": ext[1], "max": ext.max()}[cat.size_axis]
    scale = cat.size_m / current
    lo, hi = ref.bounds
    if anchor_point is None:
        center = (lo + hi) / 2
        if cat.anchor == "base":
            anchor_point = np.array([center[0], lo[1], center[2]])
        elif cat.anchor == "top":
            anchor_point = np.array([center[0], hi[1], center[2]])
        else:
            anchor_point = center
    T = trimesh.transformations.scale_matrix(scale) @ trimesh.transformations.translation_matrix(-anchor_point)
    for m in meshes:
        if m is not None:
            m.apply_transform(T)


def harmonize_texture(mesh, exclude_mask, opts):
    """
    Las caras que miran a la cámara (+Z) tienen el color real; el resto lo
    inventó SF3D. Se reasigna la luminancia del resto con la distribución
    del frente (conservando el tono) y se comprimen los brillos.
    """
    from scipy import ndimage

    from projection import _raster_triangle

    mat = mesh.visual.material
    tex = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    th, tw = tex.shape[:2]
    uvpx = np.c_[mesh.visual.uv[:, 0] * tw, (1 - mesh.visual.uv[:, 1]) * th]
    label = np.zeros((th, tw), np.uint8)          # 1 = vista, 2 = no vista
    nz = mesh.face_normals[:, 2]
    for fi in np.where(~exclude_mask)[0]:
        r = _raster_triangle(uvpx[mesh.faces[fi]], th, tw)
        if r is not None:
            xs, ys, _ = r
            label[ys, xs] = 1 if nz[fi] > opts.seen_facing else 2
    # incluir los bordes de cada isla UV (evita líneas claras por filtrado)
    grown = ndimage.grey_dilation(label, size=5)
    label = np.where(label == 0, grown, label)

    lum = tex @ np.array([0.299, 0.587, 0.114], np.float32)
    seen, unseen = label == 1, label == 2
    if seen.sum() < 500 or unseen.sum() == 0:
        return {"harmonized": False}

    ref = np.sort(lum[seen])
    # histogram matching: rango del texel no visto -> mismo cuantil del frente
    src = lum[unseen]
    ranks = np.argsort(np.argsort(src)) / max(len(src) - 1, 1)
    target = ref[(ranks * (len(ref) - 1)).astype(int)]
    out = tex.copy()
    out[unseen] *= (target / np.maximum(src, 1e-3))[:, None]

    # comprimir brillos horneados (reflejos de la foto) en todo el objeto
    obj = label > 0
    lum2 = out @ np.array([0.299, 0.587, 0.114], np.float32)
    cap = np.percentile(ref, opts.highlight_pct)
    hi = obj & (lum2 > cap)
    new_l = cap + (lum2[hi] - cap) * opts.highlight_keep
    out[hi] *= (new_l / lum2[hi])[:, None]

    mat.baseColorTexture = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))
    return {
        "harmonized": True,
        "seen_lum_median": round(float(np.median(ref)), 3),
        "unseen_lum_median_before": round(float(np.median(src)), 3),
    }


def _otsu(values, bins=256):
    hist, edges = np.histogram(values, bins=bins, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    m = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mt = m[-1]
    between = (mt * w0 - m) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return edges[np.argmax(between)]


def match_photo_colors(mesh, photo_rgba, face_mask=None, dark_only=False, chroma_keep=1.0, lum_keep=1.0, bleed=0):
    """
    Ajusta los colores de la textura a los de la foto (histogram matching por canal).
    SF3D intenta quitar la iluminación y en telas oscuras aclara demasiado el color
    (p. ej. azul marino -> azul claro). La foto es la referencia de lo que el usuario ve.
    """
    from scipy import ndimage

    from projection import _raster_triangle

    photo = np.asarray(photo_rgba.convert("RGBA")).astype(np.float32) / 255
    mask = ndimage.binary_erosion(photo[..., 3] > 0.5, iterations=3)
    if dark_only:
        # lentes: solo el marco (en la foto los cristales muestran el fondo)
        lum = photo[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
        mask &= lum < _otsu(lum[mask])
    ref = photo[..., :3][mask]
    if len(ref) < 500:
        return {"photo_matched": False}

    mat = mesh.visual.material
    tex = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    th, tw = tex.shape[:2]
    uvpx = np.c_[mesh.visual.uv[:, 0] * tw, (1 - mesh.visual.uv[:, 1]) * th]
    used = np.zeros((th, tw), bool)
    faces = mesh.faces if face_mask is None else mesh.faces[face_mask]
    for f in faces:
        r = _raster_triangle(uvpx[f], th, tw)
        if r is not None:
            used[r[1], r[0]] = True
    used = ndimage.binary_dilation(used, iterations=2)
    src = tex[used]

    before = np.median(src, 0)
    # YCbCr: Y = luminosidad, Cb/Cr = color. Ajustar por separado conserva
    # las diferencias de tono (un logo blanco sigue blanco).
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], np.float32)
    Minv = np.linalg.inv(M)
    s_ycc = src @ M.T
    r_ycc = ref @ M.T
    q = np.linspace(0, 1, 257)
    y_new = np.interp(s_ycc[:, 0], np.quantile(s_ycc[:, 0], q), np.quantile(r_ycc[:, 0], q))
    c_new = (s_ycc[:, 1:] - s_ycc[:, 1:].mean(0)) / (s_ycc[:, 1:].std(0) + 1e-4) \
        * r_ycc[:, 1:].std(0) + r_ycc[:, 1:].mean(0)
    if chroma_keep < 1.0 or lum_keep < 1.0:
        # marco de un solo color: las manchas de colores (reflejos que la reconstrucción
        # pintó en la textura) se acercan al color típico del marco en la foto
        y_med = np.median(r_ycc[:, 0])
        c_med = np.median(r_ycc[:, 1:], 0)
        y_new = y_med + (y_new - y_med) * lum_keep
        c_new = c_med + (c_new - c_med) * chroma_keep
    out = tex.copy()
    out[used] = np.c_[y_new, c_new] @ Minv.T
    if bleed > 0:
        # extender el color corregido unos píxeles fuera de cada isla UV: el visor
        # filtra la textura y, si no, mezcla el borde con colores sin corregir (puntos claros)
        near = ndimage.binary_dilation(used, iterations=bleed) & ~used
        idx = ndimage.distance_transform_edt(~used, return_distances=False, return_indices=True)
        out[near] = out[idx[0][near], idx[1][near]]
    mat.baseColorTexture = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))
    return {
        "photo_matched": True,
        "tex_median_before": [int(v * 255) for v in before],
        "tex_median_after": [int(v * 255) for v in np.median(out[used], 0)],
        "photo_median": [int(v * 255) for v in np.median(ref, 0)],
    }


def clean_texture(mesh, face_mask=None, detail_keep=0.25, strong=0.12, sigma=6.0, shading_keep=0.2):
    """
    Limpia la textura de un objeto de color casi uniforme (marco de lentes):
    color dominante + solo el detalle real. SF3D deja manchas y parches con forma
    de triángulo que, de cerca, hacen ver el modelo poligonal.
      - Detalle fuerte (cámaras, logo, bordes marcados): se conserva completo.
      - Detalle débil (manchas, ruido): se reduce a detail_keep.
      - Variación de baja frecuencia (sombras horneadas): se reduce a shading_keep.
    """
    from scipy import ndimage

    from projection import _raster_triangle

    mat = mesh.visual.material
    tex = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    th, tw = tex.shape[:2]
    uvpx = np.c_[mesh.visual.uv[:, 0] * tw, (1 - mesh.visual.uv[:, 1]) * th]
    used = np.zeros((th, tw), bool)
    faces = mesh.faces if face_mask is None else mesh.faces[face_mask]
    for f in faces:
        r = _raster_triangle(uvpx[f], th, tw)
        if r is not None:
            used[r[1], r[0]] = True
    used = ndimage.binary_dilation(used, iterations=3)
    if used.sum() < 500:
        return {"texture_cleaned": False}

    # desenfoque que solo mezcla texeles del objeto (no el fondo negro del atlas)
    w = ndimage.gaussian_filter(used.astype(np.float32), sigma)
    blur = np.stack([ndimage.gaussian_filter(tex[..., c] * used, sigma) for c in range(3)], -1)
    blur = blur / np.maximum(w, 1e-4)[..., None]
    detail = tex - blur
    base = np.median(tex[used], 0)                                   # color dominante
    strength = np.linalg.norm(detail, axis=-1)
    keep = np.where(strength > strong, 1.0, detail_keep)[..., None]
    new = base + (blur - base) * shading_keep + detail * keep
    out = tex.copy()
    out[used] = new[used]
    mat.baseColorTexture = Image.fromarray((out.clip(0, 1) * 255).astype(np.uint8))
    return {"texture_cleaned": True, "texture_base": [int(v * 255) for v in base]}


def fix_flat_depth(mesh, min_ratio):
    """
    Los modelos de una sola foto suelen subestimar la profundidad (eje de la cámara, Z).
    Si profundidad/ancho < min_ratio, se estira Z manteniendo fija la cara que vio la cámara.
    """
    ext = mesh.extents
    ratio = ext[2] / ext[0]
    if min_ratio <= 0 or ratio >= min_ratio:
        return {"depth_ratio": round(float(ratio), 2), "depth_fixed": False}
    k = min_ratio / ratio
    zfront = mesh.bounds[1, 2]
    v = mesh.vertices.copy()
    v[:, 2] = zfront - (zfront - v[:, 2]) * k
    mesh.vertices = v
    return {"depth_ratio": round(float(ratio), 2), "depth_fixed": True, "depth_scale": round(float(k), 2)}


def refine_yaw_by_symmetry(mesh, samples=60000, span_deg=30, step_deg=2):
    """
    El plano de simetría de una gorra contiene la dirección de la visera.
    Se prueba girar en Y y se elige el giro con el que el objeto es más simétrico.
    """
    from scipy.spatial import cKDTree

    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=1)
    best = (np.inf, 0.0)
    cx = np.median(pts[:, 0])
    for deg in np.arange(-span_deg, span_deg + 1e-6, step_deg):
        R = trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])[:3, :3]
        p = pts @ R.T
        c = np.median(p[:, 0])
        m = p.copy()
        m[:, 0] = 2 * c - m[:, 0]
        d, _ = cKDTree(p).query(m[::5])
        score = np.median(d)
        if score < best[0]:
            best = (score, deg)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(best[1]), [0, 1, 0]))
    return {"yaw_refine_deg": float(best[1])}


def symmetrize(mesh, samples=150000, max_asymmetry=0.05):
    """
    Gorras y lentes son simétricos izquierda-derecha. Cada vértice se promedia con
    el reflejo de la superficie: corrige el lado que la IA inventó mal.
    El plano de simetría es x = centro (tras orientar el frente hacia +Z).
    """
    from scipy.spatial import cKDTree

    pts, _ = trimesh.sample.sample_surface(mesh, samples, seed=0)
    v = mesh.vertices
    best = None
    # buscar el plano x = c que mejor refleja el objeto en sí mismo
    lo, hi = np.percentile(v[:, 0], [10, 90])
    tree = cKDTree(pts)
    for c in np.linspace((lo + hi) / 2 - 0.05 * (hi - lo), (lo + hi) / 2 + 0.05 * (hi - lo), 11):
        mirrored = v.copy()
        mirrored[:, 0] = 2 * c - mirrored[:, 0]
        d, _ = tree.query(mirrored[::7])
        score = np.median(d)
        if best is None or score < best[0]:
            best = (score, c)
    c = best[1]
    mirrored = v.copy()
    mirrored[:, 0] = 2 * c - mirrored[:, 0]
    d, idx = tree.query(mirrored)
    asym_before = float(np.median(d) / mesh.extents.max())
    if asym_before > max_asymmetry:
        # reconstrucción mala de origen: simetrizar no la arregla
        return {"asymmetry_before": round(asym_before, 4), "symmetrized": False, "quality": "low"}
    partner = pts[idx].copy()
    partner[:, 0] = 2 * c - partner[:, 0]                          # volver a reflejar
    mesh.vertices = (v + partner) / 2
    return {"asymmetry_before": round(asym_before, 4), "symmetrized": True, "quality": "ok"}


def orient_hat(mesh):
    """Gira la gorra en Y para que la visera apunte a +Z (hacia adelante)."""
    v = mesh.vertices
    y0, y1 = v[:, 1].min(), v[:, 1].max()
    h = y1 - y0
    crown = v[v[:, 1] > y0 + 0.5 * h][:, [0, 2]].mean(0)
    low = v[v[:, 1] < y0 + 0.25 * h][:, [0, 2]]
    # la visera es lo que más sobresale de la corona en la parte baja
    d = low - crown
    far = low[np.linalg.norm(d, axis=1) > np.percentile(np.linalg.norm(d, axis=1), 85)]
    direction = far.mean(0) - crown
    angle = np.arctan2(direction[0], direction[1])          # ángulo respecto de +Z
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-angle, [0, 1, 0]))
    return {"hat_rotation_deg": round(float(np.degrees(angle)), 1)}


# ---------- solo lentes ----------
def detect_lenses(mesh, cfg):
    c = mesh.triangles_center
    width = mesh.extents[0]
    front = c[:, 2] > mesh.bounds[1, 2] - cfg.front_depth_ratio * width
    lum = _face_colors(mesh) @ np.array([0.299, 0.587, 0.114])

    # 1) crecimiento de región desde las caras más claras
    adj = _face_adjacency(mesh, _welded_index(mesh))
    sel = front & (lum > cfg.lens_seed_lum)
    cand = front & (lum > cfg.lens_grow_lum)
    while True:
        new = sel.copy()
        new[adj[:, 1][sel[adj[:, 0]] & cand[adj[:, 1]]]] = True
        new[adj[:, 0][sel[adj[:, 1]] & cand[adj[:, 0]]]] = True
        if new.sum() == sel.sum():
            break
        sel = new

    # 2) relleno geométrico: lo que cae dentro del contorno de cada cristal
    #    (cubre las patillas que se veían a través del vidrio en la foto)
    cx = mesh.bounds.mean(0)[0]
    lens = sel.copy()
    for side in (c[:, 0] < cx, c[:, 0] >= cx):
        pts = c[sel & side][:, :2]
        if len(pts) < 10:
            continue
        poly = pts[ConvexHull(pts).vertices]
        center = poly.mean(0)
        hull = Path(center + (poly - center) * cfg.hull_shrink)
        inside = hull.contains_points(c[:, :2])
        lens |= front & side & inside & (lum > cfg.frame_max_lum)
    return lens


def segment_photo_lenses(photo_rgba):
    """
    Segmenta marco y cristales en la FOTO (colores reales, alta resolución).
      - Marco = píxeles oscuros del objeto (umbral de Otsu).
      - Cristales = huecos del marco, ampliados a su contorno convexo.
      - Si solo aparece un cristal (p. ej. cristal de sol tan oscuro como el marco),
        se refleja al otro lado: los lentes son simétricos.
    Devuelve (lens_masks, obj_mask) o (None, obj_mask) si no se encuentran.
    """
    from scipy import ndimage

    a = np.asarray(photo_rgba.convert("RGBA")).astype(np.float32) / 255
    obj = a[..., 3] > 0.5
    if obj.sum() < 1000:
        return None, obj
    lum = a[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    frame = obj & (lum < _otsu(lum[obj]))
    frame = ndimage.binary_closing(ndimage.binary_opening(frame, iterations=1), iterations=3)
    lab, k = ndimage.label(frame)
    if k == 0:
        return None, obj
    sizes = ndimage.sum(frame, lab, range(1, k + 1))
    frame = lab == (np.argmax(sizes) + 1)

    holes = ndimage.binary_fill_holes(frame) & ~frame
    lab, k = ndimage.label(holes)
    lenses = []
    if k:
        sizes = ndimage.sum(holes, lab, range(1, k + 1))
        h, w = frame.shape
        yy, xx = np.mgrid[0:h, 0:w]
        grid = np.c_[xx.ravel(), yy.ravel()]
        for i in np.argsort(sizes)[::-1][:2]:
            if sizes[i] < 0.05 * frame.sum():
                continue
            ys, xs = np.where(lab == i + 1)
            pts = np.c_[xs, ys]
            hull = Path(pts[ConvexHull(pts).vertices])
            lenses.append(hull.contains_points(grid).reshape(h, w))
    if not lenses:
        return None, obj
    # un solo cristal (o uno mucho menor): reflejar el mayor sobre el eje central
    if len(lenses) == 1 or lenses[1].sum() < 0.5 * lenses[0].sum():
        xs = np.where(obj.any(0))[0]
        cx = (xs.min() + xs.max()) / 2
        big = lenses[0]
        ys, xs = np.where(big)
        mx = np.round(2 * cx - xs).astype(int)
        ok = (mx >= 0) & (mx < big.shape[1])
        mirrored = np.zeros_like(big)
        mirrored[ys[ok], mx[ok]] = True
        mirrored = ndimage.binary_closing(mirrored, iterations=2)
        lenses = [big, mirrored]
    return lenses, obj


def detect_lenses_from_photo(mesh, photo_rgba, cfg):
    """
    Traslada la segmentación de la foto al modelo: se alinea la silueta frontal del
    modelo con la de la foto y cada cara frontal toma la clase del píxel al que cae.
    Funciona con cristales transparentes (se ve la mesa) y con cristales de sol.
    """
    from projection import _fit_alignment

    lenses, obj = segment_photo_lenses(photo_rgba)
    if lenses is None:
        return None, None, None
    lens_px = np.zeros_like(obj)
    for l in lenses:
        lens_px |= l
    align, iou = _fit_alignment(mesh, obj)
    sx, sy, tx, ty = align
    c = mesh.triangles_center
    width = mesh.extents[0]
    # los cristales curvos (sobre todo los de sol) llegan más atrás que el frente del aro
    front = c[:, 2] > mesh.bounds[1, 2] - cfg.lens_depth_ratio * width
    h, w = lens_px.shape
    u = np.clip((c[:, 0] * sx + tx).astype(int), 0, w - 1)
    v = np.clip((-c[:, 1] * sy + ty).astype(int), 0, h - 1)
    mask = front & lens_px[v, u]
    # si casi no cae nada en los cristales, la foto y el modelo no están alineados
    if mask.sum() < 0.05 * front.sum():
        return None, None, None
    return mask, lenses, align


def estimate_lens_tint(photo_original, photo_rgba, lenses):
    """
    Color y opacidad del cristal, comparando el cristal con el fondo de la foto.
    Un cristal transparente deja ver el fondo casi igual (ratio ~1); uno de sol, más oscuro.
    """
    if photo_original is None or not lenses:
        return None
    from scipy import ndimage

    rgb = np.asarray(photo_original.convert("RGB")).astype(np.float32) / 255
    alpha = np.asarray(photo_rgba.convert("RGBA"))[..., 3] > 128
    if rgb.shape[:2] != alpha.shape:
        return None
    lens_px = np.zeros_like(alpha)
    for l in lenses:
        lens_px |= l
    lens_px = ndimage.binary_erosion(lens_px, iterations=5)
    # fondo cercano al objeto (anillo alrededor de la silueta)
    ring = ndimage.binary_dilation(alpha, iterations=60) & ~ndimage.binary_dilation(alpha, iterations=15)
    if lens_px.sum() < 200 or ring.sum() < 200:
        return None
    w = np.array([0.299, 0.587, 0.114], np.float32)
    lens_col = np.median(rgb[lens_px], 0)
    bg_col = np.median(rgb[ring], 0)
    ratio = float((lens_col @ w) / max(bg_col @ w, 1e-3))
    # transmisión por canal: color del cristal sin el color de lo que hay detrás.
    # Lo que se ve a través suele ser la superficie bajo los lentes (mesa).
    ys, xs = np.where(alpha)
    y1, x0, x1 = ys.max(), xs.min(), xs.max()
    below = np.zeros_like(alpha)
    below[min(y1 + 20, alpha.shape[0] - 1):min(y1 + 150, alpha.shape[0]), x0:x1] = True
    ref = np.median(rgb[below], 0) if below.sum() > 200 else bg_col
    trans = lens_col / np.maximum(ref, 1e-3)
    if ratio > 0.7:                                   # transparente
        return {"color": [0.85, 0.9, 0.95], "alpha": 0.15, "ratio": round(ratio, 2), "tinted": False}
    alpha_v = float(np.clip(1.0 - ratio, 0.35, 0.85))
    return {"color": [float(v) for v in lens_col], "alpha": round(alpha_v, 2),
            "ratio": round(ratio, 2), "tinted": True,
            "transmission": [round(float(v), 3) for v in trans]}


def guess_lens_type(tint):
    """
    Sin Gemini: elige el preset por lo medido en la foto. En la foto el cristal de sol
    se ve más claro de lo que es (reflejos), así que se usa el preset oscuro de su tono.
    """
    if not tint:
        return None
    if not tint.get("tinted"):
        return "clear"
    # se clasifica por la transmisión (el color propio del cristal), no por el color
    # visto en la foto: a través del cristal se ve la mesa y todo parece amarillento
    r, g, b = tint.get("transmission") or tint["color"]
    if r > g * 1.10 and r > b * 1.15:
        return "sun_brown"
    if g > r * 1.04 and g > b * 1.03:
        return "sun_green"
    return "sun_gray"


def drop_background_parts(mesh, cfg):
    """
    Piezas que no pueden ser parte de unos lentes: separadas del cuerpo principal,
    en el centro (lejos de las bisagras) y detrás del frente. Suelen ser objetos del
    fondo de la foto que la reconstrucción tomó como parte del accesorio.
    """
    inv = _welded_index(mesh)
    adj = _face_adjacency(mesh, inv)
    lab = trimesh.graph.connected_component_labels(adj, node_count=len(mesh.faces))
    cnt = np.bincount(lab)
    main = np.argmax(cnt)
    c = mesh.triangles_center
    lo, hi = mesh.submesh([np.where(lab == main)[0]], append=True).bounds
    width = hi[0] - lo[0]
    cx = (lo[0] + hi[0]) / 2
    drop = np.zeros(len(c), bool)
    for k in np.unique(lab):
        if k == main:
            continue
        sel = lab == k
        ctr = c[sel].mean(0)
        central = abs(ctr[0] - cx) < 0.3 * width
        behind = ctr[2] < hi[2] - 0.2 * width
        if central and behind:
            drop |= sel
    return (_submesh(mesh, ~drop) if drop.any() else mesh), int(drop.sum())


def deepen_dark_color(mesh, threshold=0.30, gamma=1.35):
    """
    Marcos oscuros: en el visor los reflejos del entorno los aclaran (el negro se ve
    gris). Se oscurece un poco la textura para que el resultado se perciba como en la foto.
    """
    mat = mesh.visual.material
    if mat.baseColorTexture is None:
        return {}
    tex = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    med = float(np.median(_face_colors(mesh) @ np.array([0.299, 0.587, 0.114])))
    if med > threshold:
        return {"dark_frame": False}
    mat.baseColorTexture = Image.fromarray((np.clip(tex, 0, 1) ** gamma * 255).astype(np.uint8))
    return {"dark_frame": True}


def crop_temples(mesh, cfg):
    width = mesh.extents[0]
    zfront = mesh.bounds[1, 2]
    keep = mesh.triangles_center[:, 2] > zfront - (cfg.front_depth_ratio + cfg.temple_keep_ratio) * width
    return _submesh(mesh, keep)


def bridge_point(frame, cfg):
    c = frame.triangles_center
    width = frame.extents[0]
    cx = frame.bounds.mean(0)[0]
    front = c[:, 2] > frame.bounds[1, 2] - cfg.front_depth_ratio * width
    bridge = (np.abs(c[:, 0] - cx) < 0.05 * width) & front
    y = np.median(c[bridge, 1]) if bridge.any() else c[:, 1].mean()
    return np.array([cx, y, frame.bounds[1, 2]])


def lens_line_point(frame, lenses, cfg):
    """
    Punto de anclaje de los lentes: centro del frente a la altura de los cristales.

    - x: centro del marco (puente).
    - y: promedio de los centros de los dos cristales. Así el origen queda en la línea de
      los ojos aunque el puente esté alto (doble puente, puente "keyhole"), y la app
      solo tiene que llevarlo al puente nasal de la cara (landmark 6 de MediaPipe).
    - z: plano frontal del marco.

    Si no hay un cristal a cada lado, se usa la barra del puente (bridge_point).
    Devuelve (punto, "lens_line" | "bridge").
    """
    if lenses is not None and len(lenses.vertices):
        cx = frame.bounds.mean(0)[0]
        v = lenses.vertices
        sides = [v[v[:, 0] < cx], v[v[:, 0] >= cx]]
        if all(len(s) for s in sides):
            # centro de cada cristal = centro de su caja (no lo sesga la densidad de la malla)
            y = float(np.mean([(s[:, 1].min() + s[:, 1].max()) / 2 for s in sides]))
            return np.array([cx, y, frame.bounds[1, 2]]), "lens_line"
    return bridge_point(frame, cfg), "bridge"


def hinge_depth(frame, slice_ratio=0.018, side_ratio=0.85, max_depth_ratio=0.5):
    """
    Profundidad (z) de las bisagras: dónde termina el frente y empiezan las patillas.

    Se recorre el marco de adelante hacia atrás en franjas finas (slice_ratio del ancho).
    Mientras haya geometría cerca del centro es el frente (aro, cristales, puente). La
    primera franja en la que solo queda geometría en los extremos laterales (más allá de
    side_ratio del semiancho) es la bisagra: de ahí hacia atrás solo hay patillas.

    Devuelve la z del inicio de esa franja, o None si no se encuentra (sin patillas, o
    una forma inesperada): quien llama decide el respaldo.
    """
    v = frame.vertices
    width = frame.extents[0]
    if width <= 0 or not len(v):
        return None
    cx = frame.bounds.mean(0)[0]
    half = width / 2
    zfront = frame.bounds[1, 2]
    step = slice_ratio * width
    seen_front = False
    z = zfront
    while z > zfront - max_depth_ratio * width:
        band = v[(v[:, 2] <= z) & (v[:, 2] > z - step)]
        if len(band):
            only_sides = np.abs(band[:, 0] - cx).min() > side_ratio * half
            if only_sides and seen_front:
                return float(z)
            seen_front = seen_front or not only_sides
        z -= step
    return None


def make_ar_variant(scene, cfg, stub_ratio=0.035):
    """
    Variante para el probador de RA: el mismo modelo sin patillas.

    En la cara, las patillas quedan detrás de la cabeza y de las orejas; la app no tiene
    un oclusor que las tape, así que se ven flotando. Se corta el marco justo detrás de
    las bisagras (hinge_depth) dejando un muñón corto (stub_ratio del ancho, ~5 mm en
    lentes de 14 cm) para que la bisagra no quede abierta. Los cristales no se tocan.

    El corte se hace sobre el modelo ya normalizado: el origen, el tamaño y por lo tanto
    el ar_fit son los mismos que los del modelo completo.

    Si no se encuentra la bisagra, se usa crop_temples (corte a profundidad fija).
    Devuelve (escena, stats) con stats["method"] = "hinge" | "crop_temples".
    No modifica la escena original.
    """
    out = trimesh.Scene()
    stats = {}
    for name, geom in scene.geometry.items():
        geom = geom.copy()
        if name != "lenses":
            before = len(geom.faces)
            hinge = hinge_depth(geom)
            if hinge is not None:
                cut = hinge - stub_ratio * geom.extents[0]
                geom = _submesh(geom, geom.triangles_center[:, 2] > cut)
                stats.update(method="hinge", hinge_z_m=round(hinge, 4), cut_z_m=round(float(cut), 4))
            else:
                geom = crop_temples(geom, cfg)
                stats["method"] = "crop_temples"
            geom = remove_small_parts(geom, 0.02)    # restos sueltos de las patillas cortadas
            stats.update(faces_removed=int(before - len(geom.faces)),
                         depth_m=round(float(geom.extents[2]), 4))
        out.add_geometry(geom, node_name=name, geom_name=name)
    return out, stats


def _boundary_loops(mesh):
    """Bordes abiertos de la malla (vértices soldados), ordenados como curvas cerradas."""
    inv = _welded_index(mesh)
    fw = inv[mesh.faces]
    edges = np.sort(np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    nbrs = {}
    for a, b in bnd:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)
    loops, seen = [], set()
    for start in nbrs:
        if start in seen or len(nbrs[start]) != 2:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in nbrs[cur] if n != prev]
            if not nxt or len(nbrs[cur]) != 2:
                loop = None
                break
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            if cur in seen:
                loop = None
                break
            loop.append(cur)
            seen.add(cur)
        if loop and len(loop) >= 12:
            loops.append(np.array(loop))
    return loops, inv


def _welded_positions(mesh, inv):
    pos = np.zeros((inv.max() + 1, 3))
    pos[inv] = mesh.vertices
    return pos


def lens_rim_loops(frame, lens_centers):
    """
    Para cada centro de cristal, los vértices del borde del hueco ordenados como curva.
    Se ordenan por ángulo alrededor del centro: funciona aunque el borde tenga
    vértices "pellizcados" donde se cruzan dos tramos.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    inv = _welded_index(frame)
    fw = inv[frame.faces]
    edges = np.sort(np.concatenate([fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return [], inv
    n = inv.max() + 1
    graph = coo_matrix((np.ones(len(bnd)), (bnd[:, 0], bnd[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    bverts = np.unique(bnd)
    pos = _welded_positions(frame, inv)
    loops = []
    for c in lens_centers:
        best = None
        for k in np.unique(comp[bverts]):
            vs = bverts[comp[bverts] == k]
            if len(vs) < 12:
                continue
            pts = pos[vs]
            size = np.ptp(pts[:, :2], axis=0).max()
            d = np.linalg.norm(pts[:, :2].mean(0) - c[:2])
            if d < 0.5 * size and (best is None or d < best[0]):
                best = (d, vs)
        if best is None:
            continue
        vs = best[1]
        center = pos[vs][:, :2].mean(0)
        rel = pos[vs][:, :2] - center
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        rad = np.linalg.norm(rel, axis=1)
        # el contorno real es el más cercano al centro en cada dirección:
        # se descartan tramos sueltos que quedan más afuera (cortes, desgarros)
        bins = np.linspace(-np.pi, np.pi, 73)
        idx = np.digitize(ang, bins)
        keep = np.zeros(len(vs), bool)
        for b in np.unique(idx):
            sel = np.where(idx == b)[0]
            r_min = rad[sel].min()
            keep[sel[rad[sel] <= r_min * 1.08]] = True
        vs, ang = vs[keep], ang[keep]
        loops.append(vs[np.argsort(ang)])
    return loops, inv


def smooth_rims(frame, loops, inv, iterations=40):
    """Alisa el contorno del aro como una curva (quita el zigzag de triángulos)."""
    pos = _welded_positions(frame, inv)
    for lp in loops:
        p = pos[lp].copy()
        for _ in range(iterations):
            avg = (np.roll(p, 1, 0) + np.roll(p, -1, 0)) / 2
            p = p + 0.5 * (avg - p)
            avg = (np.roll(p, 1, 0) + np.roll(p, -1, 0)) / 2
            p = p - 0.53 * (avg - p)          # paso Taubin: evita que el contorno se encoja
        pos[lp] = p
    frame.vertices = pos[inv]
    return frame


def build_lenses_from_rims(frame, loops, inv, cfg, tint=None):
    """Cristal plano que calza exacto con el contorno del aro (sin ondas de SF3D)."""
    pos = _welded_positions(frame, inv)
    meshes = []
    for lp in loops:
        pts = pos[lp]
        ctr = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - ctr)
        normal = vt[2]
        flat = pts - np.outer((pts - ctr) @ normal, normal)     # proyectar al plano
        inner = ctr + (flat - ctr) * 0.5                          # anillo intermedio
        verts = np.vstack([ctr, inner, flat])
        n = len(flat)
        i = np.arange(n)
        j = (i + 1) % n
        faces = np.vstack([
            np.c_[np.zeros(n, int), 1 + i, 1 + j],                # centro -> anillo interno
            np.c_[1 + i, 1 + n + i, 1 + n + j],                   # anillo interno -> borde
            np.c_[1 + i, 1 + n + j, 1 + j],
        ])
        if normal[2] < 0:
            normal = -normal
        # el cristal va encajado dentro del aro, un poco detrás del frente
        verts = verts - normal * cfg.lens_inset_ratio * frame.extents[0]
        m = trimesh.Trimesh(verts, faces, process=False)
        if m.face_normals[:, 2].mean() < 0:                     # que mire al frente
            m.faces = m.faces[:, ::-1]
        meshes.append(m)
    if not meshes:
        return None
    lenses = trimesh.util.concatenate(meshes)
    lenses.visual = trimesh.visual.TextureVisuals(material=lens_material(cfg, tint))
    return lenses


def remove_small_parts(mesh, min_ratio=0.10):
    """Elimina piezas sueltas diminutas (ruido de la reconstrucción)."""
    inv = _welded_index(mesh)
    welded = trimesh.Trimesh(np.zeros((inv.max() + 1, 3)), inv[mesh.faces], process=False)
    labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    keep = counts[labels] >= min_ratio * counts.max()          # relativo a la pieza principal
    return _submesh(mesh, keep) if not keep.all() else mesh


def lens_material(cfg, tint=None):
    color = tint["color"] if tint else [0.85, 0.9, 0.95]
    alpha = tint["alpha"] if tint else cfg.lens_alpha
    return PBRMaterial(
        name="lens",
        baseColorFactor=list(color) + [alpha],
        metallicFactor=float(tint.get("metallic", 0.0)) if tint else 0.0,
        roughnessFactor=float(tint.get("roughness", 0.05)) if tint else 0.05,
        alphaMode="BLEND",
        doubleSided=True,
    )


def export_glb(scene, path, unlit=False, specular=None):
    """
    Exporta a .glb.
    - unlit=True: KHR_materials_unlit en los materiales opacos (colores exactos, sin luz).
    - specular=0.5: KHR_materials_specular en el marco de plástico (menos reflejo del
      entorno, que hace ver gris el negro). Los visores que no lo soportan lo ignoran.
    """
    import json
    import struct

    data = scene.export(file_type="glb", include_normals=True)
    if unlit or specular is not None:
        json_len = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20 + json_len])
        used = set(gltf.get("extensionsUsed", []))
        for m in gltf.get("materials", []):
            if m.get("alphaMode") == "BLEND":          # los cristales mantienen su brillo
                continue
            if unlit:
                m.setdefault("extensions", {})["KHR_materials_unlit"] = {}
                used.add("KHR_materials_unlit")
            elif specular is not None:
                m.setdefault("extensions", {})["KHR_materials_specular"] = {"specularFactor": float(specular)}
                used.add("KHR_materials_specular")
        gltf["extensionsUsed"] = sorted(used)
        js = json.dumps(gltf, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        rest = data[20 + json_len:]
        total = 12 + 8 + len(js) + len(rest)
        data = struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON") + js + rest
    with open(path, "wb") as f:
        f.write(data)


# ---------- modo ligero (SPAR3D) ----------
def clean_face_mask(mesh, mask, passes=3, keep_parts=6):
    """
    Limpia una máscara por caras: voto de mayoría entre vecinas (quita puntos sueltos
    y dientes en el borde) y deja solo las piezas más grandes (un cristal por lado).
    """
    inv = _welded_index(mesh)
    adj = _face_adjacency(mesh, inv)
    n = len(mesh.faces)
    A = sp.coo_matrix((np.ones(2 * len(adj)), (np.r_[adj[:, 0], adj[:, 1]], np.r_[adj[:, 1], adj[:, 0]])),
                      shape=(n, n)).tocsr()
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    m = mask.copy()
    for _ in range(passes):
        votes = (A @ m.astype(float) + m) / (deg + 1)
        m = votes > 0.5
    if m.any() and keep_parts:
        sel = np.where(m)[0]
        sub = adj[m[adj[:, 0]] & m[adj[:, 1]]]
        remap = -np.ones(n, int)
        remap[sel] = np.arange(len(sel))
        lab = trimesh.graph.connected_component_labels(remap[sub], node_count=len(sel))
        cnt = np.bincount(lab)
        big = np.argsort(-cnt)[:keep_parts]
        big = big[cnt[big] >= 0.15 * cnt.max()]
        m[:] = False
        m[sel[np.isin(lab, big)]] = True
    return m


def fill_lens_holes(mesh, mask, cfg, shrink=1.0, allowed=None):
    """
    Rellena huecos dentro de cada cristal (reflejos o sombras mal clasificados):
    toda cara del frente cuyo centro cae dentro del contorno del cristal, visto de
    frente y contraído un poco para no tocar el aro, pasa a ser cristal.
    """
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull

    if mask.sum() < 20:
        return mask
    c = mesh.triangles_center
    front = c[:, 2] > mesh.bounds[1, 2] - cfg.lens_depth_ratio * mesh.extents[0]
    cx = mesh.bounds.mean(0)[0]
    out = mask.copy()
    for side in (c[:, 0] < cx, c[:, 0] >= cx):
        pts = c[mask & side][:, :2]
        if len(pts) < 10:
            continue
        hull = pts[ConvexHull(pts).vertices]
        center = hull.mean(0)
        path = Path(center + (hull - center) * shrink)
        add = front & side & path.contains_points(c[:, :2])
        if allowed is not None:
            add &= allowed
        out |= add
    return out


def absorb_islands(mesh, mask, cfg, min_ratio=0.02):
    """
    Islas pequeñas que no son cristal pero quedaron rodeadas por él (reflejos oscuros,
    restos del interior): si están en el frente pasan a ser cristal. Si no, al separar
    el marco se borrarían y quedaría un hueco en la luna.
    """
    inv = _welded_index(mesh)
    adj = _face_adjacency(mesh, inv)
    rest = ~mask
    sub = adj[rest[adj[:, 0]] & rest[adj[:, 1]]]
    lab = trimesh.graph.connected_component_labels(sub, node_count=len(mask))
    cnt = np.bincount(lab, weights=rest.astype(float))
    small = rest & (cnt[lab] < min_ratio * cnt.max())
    c = mesh.triangles_center
    front = c[:, 2] > mesh.bounds[1, 2] - cfg.lens_depth_ratio * mesh.extents[0]
    return mask | (small & front)


def _lowpass(r, harmonics):
    spec = np.fft.rfft(r)
    spec[harmonics + 1:] = 0
    return np.fft.irfft(spec, len(r))


def _radial_profile(pts, center, grid, bins, stat):
    rel = pts - center
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    dist = np.linalg.norm(rel, axis=1)
    idx = ((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    prof = np.full(bins, np.nan)
    for b in range(bins):
        d = dist[idx == b]
        if len(d) >= 3:
            prof[b] = stat(d)
    ok = ~np.isnan(prof)
    if ok.sum() < bins // 2:
        return None
    centers = -np.pi + (np.arange(bins) + 0.5) * 2 * np.pi / bins
    return np.interp(grid, centers[ok], prof[ok], period=2 * np.pi)


def _smooth_outline(xy, frame_xy=None, n=128, harmonics=8, grow=1.02, outer_xy=None, rim_ratio=0.90):
    """
    Contorno del cristal visto de frente, como curva suave: radio en función del
    ángulo (desde el centro) filtrado con Fourier. Quita picos y dientes del borde.
    Si se pasa el marco, el contorno no puede pasar el borde interior del aro.
    """
    from scipy.spatial import ConvexHull

    hull = xy[ConvexHull(xy).vertices]
    center = hull.mean(0)
    rel = hull - center
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(ang)
    ang, rad = ang[order], np.linalg.norm(rel[order], axis=1)
    grid = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = np.interp(grid, ang, rad, period=2 * np.pi)

    if frame_xy is not None and len(frame_xy) > 50:
        # borde interior del aro: radio del marco más cercano en cada dirección
        # (percentil bajo: ignora aletas sueltas que se meten hacia el centro)
        fr = frame_xy - center
        fa = np.arctan2(fr[:, 1], fr[:, 0])
        fd = np.linalg.norm(fr, axis=1)
        bins = 64
        idx = ((fa + np.pi) / (2 * np.pi) * bins).astype(int) % bins
        rim = np.full(bins, np.nan)
        for b in range(bins):
            d = fd[(idx == b) & (fd > 0.25 * np.median(r))]
            if len(d) >= 3:
                rim[b] = np.percentile(d, 15)
        ok = ~np.isnan(rim)
        if ok.sum() >= bins // 2:
            centers = -np.pi + (np.arange(bins) + 0.5) * 2 * np.pi / bins
            rim_r = np.interp(grid, centers[ok], rim[ok], period=2 * np.pi)
            r = np.minimum(r, _lowpass(rim_r, harmonics) * 1.03)

    if outer_xy is not None and len(outer_xy) > 50:
        # nunca más allá del contorno exterior del lente menos el grosor del aro
        outer = _radial_profile(outer_xy, center, grid, 64, lambda d: np.percentile(d, 97))
        if outer is not None:
            r = np.minimum(r, _lowpass(outer, harmonics) * rim_ratio / grow)

    r = _lowpass(r, harmonics) * grow
    return center, np.c_[np.cos(grid), np.sin(grid)] * r[:, None]


def _fit_quadric(pts):
    """z = a + bx + cy + dx² + ey² + fxy, robusto (descarta puntos lejanos y repite)."""
    keep = np.ones(len(pts), bool)
    coef = None
    for _ in range(3):
        x, y, z = pts[keep].T
        A = np.c_[np.ones_like(x), x, y, x * x, y * y, x * y]
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        x, y, z = pts.T
        res = z - np.c_[np.ones_like(x), x, y, x * x, y * y, x * y] @ coef
        keep = np.abs(res - np.median(res)) < 2.5 * (np.median(np.abs(res - np.median(res))) + 1e-9)
    return lambda x, y: np.c_[np.ones_like(x), x, y, x * x, y * y, x * y] @ coef


def build_lens_surface(mesh, mask, tint, cfg, rings=24):
    """
    Cristal nuevo y perfectamente liso para cada lado.
    - Contorno: el del cristal de SPAR3D visto de frente, suavizado (sin dientes).
    - Curvatura: una superficie cuadrática ajustada a los puntos del cristal original
      (lentes reales son superficies suaves; la malla de SPAR3D trae ondas y huecos).
    - Malla en anillos concéntricos: triángulos regulares, reflejos limpios.
    """
    parts = []
    c = mesh.triangles_center
    cx = mesh.bounds.mean(0)[0]
    for side in (c[:, 0] < cx, c[:, 0] >= cx):
        sel = mask & side
        if sel.sum() < 20:
            continue
        pts = mesh.vertices[np.unique(mesh.faces[sel])]
        # un solo z por posición (el más adelante): evita mezclar cara frontal y trasera
        key = np.round(pts[:, :2] / (mesh.extents[0] * 0.004)).astype(int)
        order = np.lexsort((-pts[:, 2], key[:, 1], key[:, 0]))
        _, first = np.unique(key[order], axis=0, return_index=True)
        pts = pts[order[first]]
        if len(pts) < 10:
            continue
        front = c[:, 2] > mesh.bounds[1, 2] - cfg.lens_depth_ratio * mesh.extents[0]
        frame_xy = c[~mask & front & side][:, :2]
        center, outline = _smooth_outline(pts[:, :2], frame_xy, outer_xy=c[front & side][:, :2])
        surf = _fit_quadric(pts)
        n = len(outline)
        verts = [center[None]]
        for k in range(1, rings + 1):
            verts.append(center + outline * (k / rings))
        xy = np.vstack(verts)
        v = np.c_[xy, surf(xy[:, 0], xy[:, 1])]
        faces = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
        for k in range(1, rings):
            a0, b0 = 1 + (k - 1) * n, 1 + k * n
            for i in range(n):
                j = (i + 1) % n
                faces += [[a0 + i, b0 + i, b0 + j], [a0 + i, b0 + j, a0 + j]]
        part = trimesh.Trimesh(v, np.array(faces), process=False)
        part.metadata["outline"] = center + outline / 1.02      # contorno sin el margen
        fix = part.face_normals[:, 2] < 0            # todas mirando al frente
        part.faces[fix] = part.faces[fix][:, ::-1]
        parts.append(part)
    if not parts:
        return None
    outlines = [p.metadata["outline"] for p in parts]
    lenses = trimesh.util.concatenate(parts)
    lenses.metadata["outlines"] = outlines
    lenses.visual = trimesh.visual.TextureVisuals(material=lens_material(cfg, tint))
    return lenses


def lenses_from_holes(frame, cfg):
    """
    Marcos transparentes: la reconstrucción deja huecos donde están los cristales.
    Cada hueco grande del frente (uno por lado) se tapa con un cristal liso.
    """
    c = frame.triangles_center
    cx = frame.bounds.mean(0)[0]
    centers = [c[side].mean(0) for side in (c[:, 0] < cx, c[:, 0] >= cx) if side.any()]
    loops, inv = lens_rim_loops(frame, centers)
    pos = _welded_positions(frame, inv)
    parts, outlines = [], []
    for lp in loops:
        pts = pos[lp]
        if len(pts) < 12 or np.ptp(pts[:, 0]) < 0.15 * frame.extents[0]:
            continue                                   # hueco chico: no es un cristal
        center, outline = _smooth_outline(pts[:, :2], grow=1.04)
        surf = _fit_quadric(pts)
        n, rings = len(outline), 16
        xy = np.vstack([center[None]] + [center + outline * (k / rings) for k in range(1, rings + 1)])
        v = np.c_[xy, surf(xy[:, 0], xy[:, 1])]
        faces = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
        for k in range(1, rings):
            a0, b0 = 1 + (k - 1) * n, 1 + k * n
            for i in range(n):
                j = (i + 1) % n
                faces += [[a0 + i, b0 + i, b0 + j], [a0 + i, b0 + j, a0 + j]]
        part = trimesh.Trimesh(v, np.array(faces), process=False)
        fix = part.face_normals[:, 2] < 0
        part.faces[fix] = part.faces[fix][:, ::-1]
        parts.append(part)
        outlines.append(center + outline)
    if not parts:
        # sin huecos claros: un cristal por lado dentro del frente (tapa huecos sueltos)
        front = c[:, 2] > frame.bounds[1, 2] - cfg.lens_depth_ratio * frame.extents[0]
        lenses = build_lens_surface(frame, front, None, cfg, rings=16)
        if lenses is None:
            return None
        v = lenses.vertices.copy()
        for side in (v[:, 0] < cx, v[:, 0] >= cx):
            ctr = v[side, :2].mean(0)
            v[side, :2] = ctr + (v[side, :2] - ctr) * 0.85
        lenses.vertices = v
        return lenses
    lenses = trimesh.util.concatenate(parts)
    lenses.metadata["outlines"] = outlines
    lenses.visual = trimesh.visual.TextureVisuals(material=lens_material(cfg, None))
    return lenses


def trim_rim_flaps(main, outlines, cfg, shrink=0.95):
    """
    Quita las "aletas" del marco que se meten dentro del cristal (bordes grumosos
    del aro): caras del frente cuyo centro cae dentro del contorno liso del cristal.
    """
    from matplotlib.path import Path

    c = main.triangles_center
    front = c[:, 2] > main.bounds[1, 2] - cfg.lens_depth_ratio * main.extents[0]
    inside = np.zeros(len(c), bool)
    for ol in outlines:
        ctr = ol.mean(0)
        inside |= Path(ctr + (ol - ctr) * shrink).contains_points(c[:, :2])
    drop = inside & front
    return (_submesh(main, ~drop) if drop.any() else main), int(drop.sum())


def _dark_faces(mesh, lens_mask):
    """Caras tan oscuras como el marco típico (el aro), por textura."""
    lum = _face_colors(mesh) @ np.array([0.299, 0.587, 0.114])
    rest = ~lens_mask
    ref = lum[rest] if rest.sum() > 50 else lum
    return lum <= np.percentile(ref, 60)


def tame_bright_spots(main, pct=92):
    """
    Manchas muy claras en el marco (reflejos de la foto o restos del cristal en el
    borde del aro): esas caras toman el color típico del marco.
    """
    from projection import _raster_triangle

    mat = main.visual.material
    if mat.baseColorTexture is None or len(main.faces) < 100:
        return {}
    tex = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32)
    lum = _face_colors(main) @ np.array([0.299, 0.587, 0.114])
    thr = np.percentile(lum, pct)
    typical = np.percentile(lum, 50)
    if thr - typical < 0.15:                         # marco uniforme: nada que corregir
        return {"bright_spots": 0}
    bright = lum > thr
    color = np.median(_face_colors(main)[~bright], 0) * 255
    h, w = tex.shape[:2]
    uvpx = np.c_[main.visual.uv[:, 0] * w, (1 - main.visual.uv[:, 1]) * h]
    for f in main.faces[bright]:
        r = _raster_triangle(uvpx[f], h, w)
        if r is not None:
            tex[r[1], r[0]] = color
    mat.baseColorTexture = Image.fromarray(tex.clip(0, 255).astype(np.uint8))
    return {"bright_spots": int(bright.sum())}


def apply_frame_preset(main, opts, cat):
    """Rugosidad y relieve de la biblioteca (el color sigue siendo el de la foto)."""
    try:
        from material_presets import bake_frame_material, frame_preset
        if opts.frame_material == "off":
            return {"frame_material": "none"}
        name, preset = frame_preset(opts.frame_material, cat.name)
        if not preset:
            return {}
        mat = main.visual.material
        size = mat.baseColorTexture.size[0] if mat.baseColorTexture is not None else 1024
        baked = bake_frame_material(main, preset, size=size)
        if baked is None:
            return {}
        mr, normal, detail = baked
        mat.metallicRoughnessTexture = mr
        mat.metallicFactor = 1.0          # los valores reales van en la textura
        mat.roughnessFactor = 1.0
        if normal is not None:
            mat.normalTexture = normal
        if detail is not None and mat.baseColorTexture is not None:
            base = np.asarray(mat.baseColorTexture.convert("RGB").resize((size, size))).astype(np.float32)
            mat.baseColorTexture = Image.fromarray(np.clip(base * detail[..., None], 0, 255).astype(np.uint8))
        return {"frame_material": name}
    except Exception as exc:              # la biblioteca es un extra: nunca tumba la generación
        print(f"[mirage-ai-3d] material de marco omitido: {exc}")
        return {"frame_material": "none"}


def _subdivide_with_mask(mesh, mask, max_faces, smooth_iterations):
    """Subdivide (hasta 2 niveles) y propaga la máscara: cada cara se vuelve 4 hijas."""
    for _ in range(2):
        if len(mesh.faces) * 4 > max_faces:
            break
        mesh, _ = subdivide_textured(mesh, max_faces=10 ** 9, smooth_iterations=smooth_iterations)
        if mask is not None:
            mask = np.tile(mask, 4)
    return mesh, mask


def process_light(mesh, photo_rgba, opts, cat, photo_original, stats):
    """
    SPAR3D ya entrega una malla completa (aro, cristales y patillas). Aquí no se
    reconstruye nada: se suaviza todo junto (sin grietas entre piezas), se separan
    los cristales con material transparente y se ajusta tamaño y color.
    """
    src_mat = mesh.visual.material
    mesh = orient_front_to_positive_z(mesh, cat.name)
    stats["mode"] = "light"

    lens_mask = lens_tint = None
    if cat.name == "glasses":
        mesh, stats["background_faces_removed"] = drop_background_parts(mesh, opts.glasses)
        photo_lenses = None
        if photo_rgba is not None:
            lens_mask, photo_lenses, _ = detect_lenses_from_photo(mesh, photo_rgba, opts.glasses)
        # SPAR3D pinta claros los cristales transparentes: la textura los detecta bien.
        # La foto aporta los cristales oscuros (de sol). Se usan ambas.
        tex_mask = detect_lenses(mesh, opts.glasses)
        if lens_mask is None:
            lens_mask, stats["lens_source"] = tex_mask, "texture"
        else:
            # la textura completa lo que la foto no alcanzó (fotos en ángulo), pero nunca
            # en el puente: ahí lo claro es metal, no cristal
            c = mesh.triangles_center
            cx = mesh.bounds.mean(0)[0]
            bridge = np.abs(c[:, 0] - cx) < 0.08 * mesh.extents[0]
            lens_mask, stats["lens_source"] = lens_mask | (tex_mask & ~bridge), "photo+texture"
        lens_tint = estimate_lens_tint(photo_original, photo_rgba, photo_lenses)
        if opts.use_material_library:
            from material_presets import lens_preset
            lens_type = opts.lens_type or guess_lens_type(lens_tint)
            preset = lens_preset(lens_type) if lens_type else None
            if preset:
                lens_tint = dict(preset, preset=lens_type,
                                 source="gemini" if opts.lens_type else "photo")
        allowed = None
        lum = _face_colors(mesh) @ np.array([0.299, 0.587, 0.114])
        if 50 < lens_mask.sum() < len(lens_mask) - 50:
            # si en la textura el cristal es más claro que el marco, las caras oscuras
            # dentro de la máscara son el aro y no se deben volver cristal
            if np.median(lum[lens_mask]) > np.median(lum[~lens_mask]) + 0.12:
                thr = np.percentile(lum[~lens_mask], 60)
                allowed = lum > thr
                lens_mask &= allowed
        lens_mask = clean_face_mask(mesh, lens_mask)
        lens_mask = fill_lens_holes(mesh, lens_mask, opts.glasses, allowed=allowed)
        lens_mask = clean_face_mask(mesh, lens_mask)
        lens_mask = absorb_islands(mesh, lens_mask, opts.glasses)
        if lens_tint:
            stats["lens_tint"] = lens_tint
        if not lens_mask.any():
            lens_mask = None

    clear_frame = False
    if lens_mask is not None and stats.get("lens_source") == "texture":
        c = mesh.triangles_center
        front = c[:, 2] > mesh.bounds[1, 2] - opts.glasses.lens_depth_ratio * mesh.extents[0]
        if (lens_mask & front).sum() > 0.6 * front.sum():
            # todo el frente parece cristal: marco transparente (acetato cristal).
            # No se separan lunas; todo el lente se vuelve material translúcido.
            clear_frame, lens_mask = True, None
            stats["clear_frame"] = True

    if cat.match_photo and photo_rgba is not None:
        if lens_mask is not None:
            stats.update(match_photo_colors(mesh, photo_rgba, face_mask=~lens_mask, dark_only=True,
                                            chroma_keep=opts.frame_chroma_keep, lum_keep=opts.frame_lum_keep,
                                            bleed=8))
        else:
            stats.update(match_photo_colors(mesh, photo_rgba))

    if not opts.unlit and not clear_frame:
        face_ok = ~lens_mask if lens_mask is not None else np.ones(len(mesh.faces), bool)
        sub = _submesh(mesh, face_ok)
        sub.visual.material = mesh.visual.material
        if float(np.median(_face_colors(sub) @ np.array([0.299, 0.587, 0.114]))) <= 0.15:   # solo negros
            stats.update(deepen_dark_color(mesh, threshold=1.0))

    if cat.min_depth_ratio > 0:
        stats.update(fix_flat_depth(mesh, cat.min_depth_ratio))
    if cat.name == "hat":
        stats.update(orient_hat(mesh))

    # suavizar y subdividir la malla entera: aro y cristales comparten borde
    mesh = taubin_smooth(mesh, cat.smooth_iterations * opts.light_smooth_factor)
    mesh, lens_mask = _subdivide_with_mask(mesh, lens_mask, opts.max_faces, smooth_iterations=opts.light_subdiv_smooth)
    stats["subdivided"] = True

    lenses = None
    if lens_mask is not None:
        # restos del marco dentro del cristal (se verían como manchas a través de él)
        inner = fill_lens_holes(mesh, lens_mask, opts.glasses, shrink=0.80) & ~lens_mask
        inner &= ~_dark_faces(mesh, lens_mask)               # el aro (oscuro) nunca se borra
        stats["lens_debris_removed"] = int(inner.sum())
        main = _submesh(mesh, ~(lens_mask | inner))
        lenses = build_lens_surface(mesh, lens_mask, lens_tint, opts.glasses)
        if lenses is None:                                   # respaldo: la malla tal cual
            lenses = _submesh(mesh, lens_mask)
            lenses.visual = trimesh.visual.TextureVisuals(material=lens_material(opts.glasses, lens_tint))
        stats["lens_faces"] = int(len(lenses.faces))
        # patillas: SPAR3D las genera bien, se conservan (en RA las tapa la cabeza)
        if "outlines" in lenses.metadata:
            main, stats["rim_flaps_removed"] = trim_rim_flaps(main, lenses.metadata["outlines"], opts.glasses)
        # borde interior del aro: alisar el zigzag de triángulos
        c = lenses.triangles_center
        cx = mesh.bounds.mean(0)[0]
        centers = [c[side].mean(0) for side in (c[:, 0] < cx, c[:, 0] >= cx) if side.any()]
        loops, inv = lens_rim_loops(main, centers)
        if loops:
            main = smooth_rims(main, loops, inv, iterations=30)
        stats["rims_smoothed"] = len(loops)
    else:
        main = mesh
        if clear_frame:
            lenses = lenses_from_holes(main, opts.glasses)
            stats["lens_holes_filled"] = 0 if lenses is None else len(lenses.metadata.get("outlines", []))
    main = remove_small_parts(main, 0.02)
    # material con la textura ya corregida (no el original: al quitar piezas del
    # fondo la malla se copia y el material original queda sin las correcciones)
    apply_material(main, mesh.visual.material, cat, opts)
    if lenses is not None and not clear_frame:
        stats.update(tame_bright_spots(main))

    anchor = None
    if cat.anchor == "bridge":
        anchor, stats["anchor"] = lens_line_point(main, lenses, opts.glasses)
    normalize([main, lenses], cat, anchor)
    if clear_frame:
        mat = main.visual.material
        if mat.baseColorTexture is not None:
            # en la foto, a través del acetato se ve la mesa: se aclara hacia blanco
            t = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
            gray = t @ np.array([0.299, 0.587, 0.114], np.float32)
            t = 0.5 * t + 0.5 * gray[..., None]           # menos tono de la mesa
            t = 0.45 * t + 0.55                           # hacia blanco
            mat.baseColorTexture = Image.fromarray((t.clip(0, 1) * 255).astype(np.uint8))
        mat.alphaMode = "BLEND"
        mat.baseColorFactor = [1.0, 1.0, 1.0, opts.clear_frame_alpha]
        mat.roughnessFactor = 0.1
        mat.metallicFactor = 0.0
        stats["frame_material"] = "clear_plastic"
    elif opts.use_material_library and not opts.unlit:
        stats.update(apply_frame_preset(main, opts, cat))
    smooth_normals(main, diffuse=opts.normal_diffuse)
    if lenses is not None:
        smooth_normals(lenses)

    out = trimesh.Scene()
    out.add_geometry(main, node_name=cat.name, geom_name=cat.name)
    if lenses is not None:
        out.add_geometry(lenses, node_name="lenses", geom_name="lenses")
    stats.update({"faces": int(len(main.faces)), "size_m": [round(float(v), 4) for v in main.extents]})
    return out, stats


# ---------- punto de entrada ----------
def process_accessory(scene_or_mesh, photo_rgba=None, opts=PostprocessOptions(), photo_original=None):
    if opts.category not in CATEGORIES:
        raise ValueError(f"Categoría no soportada: {opts.category}")
    cat = CATEGORIES[opts.category]
    if opts.mode == "light":
        mesh = _to_mesh(scene_or_mesh)
        mesh.visual.material.baseColorTexture = _limit(mesh.visual.material.baseColorTexture, opts.max_texture_px)
        return process_light(mesh, photo_rgba, opts, cat, photo_original, {"category": cat.name})
    mesh = _to_mesh(scene_or_mesh)
    src_mat = mesh.visual.material
    src_mat.baseColorTexture = _limit(src_mat.baseColorTexture, opts.max_texture_px)
    stats = {"category": cat.name}

    mesh = orient_front_to_positive_z(mesh, cat.name)

    lens_mask = None
    lens_tint = None
    if cat.name == "glasses":
        photo_lenses = photo_align = None
        if photo_rgba is not None:
            # 1) desde la foto: funciona con cristales transparentes y de sol
            lens_mask, photo_lenses, photo_align = detect_lenses_from_photo(mesh, photo_rgba, opts.glasses)
        stats["lens_source"] = "photo" if lens_mask is not None else "texture"
        if lens_mask is None:
            # 2) respaldo: por color de la textura (cristales claros)
            lens_mask = detect_lenses(mesh, opts.glasses)
        lens_tint = estimate_lens_tint(photo_original, photo_rgba, photo_lenses)
        if lens_tint:
            stats["lens_tint"] = lens_tint

    if opts.harmonize_colors:
        excl = lens_mask if lens_mask is not None else np.zeros(len(mesh.faces), bool)
        stats.update(harmonize_texture(mesh, excl, opts))

    if cat.match_photo and photo_rgba is not None:
        if lens_mask is not None:
            stats.update(match_photo_colors(mesh, photo_rgba, face_mask=~lens_mask, dark_only=True))
        else:
            stats.update(match_photo_colors(mesh, photo_rgba))

    if lens_mask is not None:
        stats.update(clean_texture(mesh, face_mask=~lens_mask))
        # foto real sobre el frente del marco (después de limpiar, para conservar su detalle)
        if opts.project_photo and photo_align is not None:
            from projection import project_photo_frame
            stats.update(project_photo_frame(mesh, photo_rgba, ~lens_mask, photo_lenses, photo_align))

    if cat.min_depth_ratio > 0:
        stats.update(fix_flat_depth(mesh, cat.min_depth_ratio))

    if cat.name == "hat":
        stats.update(orient_hat(mesh))
        stats.update(refine_yaw_by_symmetry(mesh))
        stats.update(symmetrize(mesh))

    if opts.project_photo and photo_rgba is not None and cat.name != "glasses":
        from projection import project_photo        # versión antigua (experimental)
        stats.update(project_photo(mesh, photo_rgba))

    lenses = None
    if lens_mask is not None:
        # centros de los cristales (uno por lado) antes de separarlos
        c = mesh.triangles_center
        cx = mesh.bounds.mean(0)[0]
        lens_centers = [c[lens_mask & side].mean(0) for side in (c[:, 0] < cx, c[:, 0] >= cx)
                        if (lens_mask & side).any()]
        main = _submesh(mesh, ~lens_mask)
        if not opts.glasses.keep_temples:
            main = crop_temples(main, opts.glasses)
            main = remove_small_parts(main, 0.10)    # restos de patillas cortadas
        else:
            main = remove_small_parts(main, 0.02)    # las patillas pueden ser piezas aparte
        stats["lens_faces"] = int(lens_mask.sum())

        # 1) alisar el contorno del aro (quita el zigzag de triángulos)
        loops, inv = lens_rim_loops(main, lens_centers)
        main = smooth_rims(main, loops, inv)
        # 2) superficie: suavizar, subdividir x2 y volver a suavizar (curvas en vez de facetas)
        main = taubin_smooth(main, cat.smooth_iterations)
        main, subdivided = subdivide_textured(main, opts.max_faces, smooth_iterations=10, levels=2)
        # 3) cristales planos que calzan exacto con el aro ya suavizado
        loops, inv = lens_rim_loops(main, lens_centers)
        lenses = build_lenses_from_rims(main, loops, inv, opts.glasses, lens_tint)
        if lenses is None and lens_mask.any():                    # respaldo
            lenses = taubin_smooth(_submesh(mesh, lens_mask), cat.smooth_iterations * 2)
            lenses.visual = trimesh.visual.TextureVisuals(material=lens_material(opts.glasses, lens_tint))
        stats["lens_rims"] = len(loops)
    else:
        main = taubin_smooth(mesh, cat.smooth_iterations)
        main, subdivided = subdivide_textured(main, opts.max_faces)

    stats["subdivided"] = subdivided
    apply_material(main, src_mat, cat, opts)

    anchor = None
    if cat.anchor == "bridge":
        anchor, stats["anchor"] = lens_line_point(main, lenses, opts.glasses)
    normalize([main, lenses], cat, anchor)

    # normales al final (después de escalar y rotar)
    smooth_normals(main)
    if lenses is not None:
        smooth_normals(lenses)

    out = trimesh.Scene()
    out.add_geometry(main, node_name=cat.name, geom_name=cat.name)
    if lenses is not None:
        out.add_geometry(lenses, node_name="lenses", geom_name="lenses")
    stats.update({
        "faces": int(len(main.faces)),
        "size_m": [round(float(v), 4) for v in main.extents],
    })
    return out, stats
