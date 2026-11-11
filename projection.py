"""
Proyección de la foto original sobre la textura del modelo.

SF3D genera una textura borrosa y con colores lavados. La foto del usuario
tiene el color real y más resolución. Este módulo "pega" la foto sobre las
caras que miran a la cámara (+Z), como si fuera un proyector.

Pasos:
  1. Alinear: busca la escala y el desplazamiento que mejor hacen coincidir
     la silueta del modelo (vista frontal) con la máscara de la foto.
  2. Visibilidad: z-buffer para no pintar caras tapadas por otras.
  3. Pintar: recorre cada triángulo en el atlas UV y mezcla el color de la
     foto según cuánto mira la cara hacia el frente.

Supuesto: el modelo ya está orientado con el frente hacia +Z y la foto es
frontal (la misma que se usó para generar el modelo).
"""
import numpy as np
from PIL import Image


# ---------- rasterizado básico ----------
def _raster_triangle(p, h, w):
    """Píxeles dentro del triángulo p (3x2, coords de píxel) y sus baricéntricas."""
    x0, y0 = np.floor(p.min(0)).astype(int)
    x1, y1 = np.ceil(p.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w - 1), min(y1, h - 1)
    if x1 < x0 or y1 < y0:
        return None
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    px = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
    a, b, c = p
    v0, v1 = b - a, c - a
    d = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(d) < 1e-12:
        return None
    r = px - a
    l1 = (r[:, 0] * v1[1] - v1[0] * r[:, 1]) / d
    l2 = (v0[0] * r[:, 1] - r[:, 0] * v0[1]) / d
    l0 = 1 - l1 - l2
    inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
    if not inside.any():
        return None
    bary = np.stack([l0, l1, l2], 1)[inside]
    return xs.ravel()[inside], ys.ravel()[inside], bary


def _zbuffer(xy, z, faces, h, w):
    """Vista frontal: guarda la z más cercana (mayor z) por píxel."""
    zbuf = np.full((h, w), -np.inf)
    for f in faces:
        r = _raster_triangle(xy[f], h, w)
        if r is None:
            continue
        xs, ys, bary = r
        zz = bary @ z[f]
        np.maximum.at(zbuf, (ys, xs), zz)
    return zbuf


# ---------- alineación foto <-> modelo ----------
def _bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def _fit_alignment(mesh, photo_mask, grid=192):
    """Devuelve ((sx, sy, tx, ty), iou): pixel = (x*sx + tx, -y*sy + ty)."""
    v = mesh.vertices
    xmin, ymin = v[:, 0].min(), v[:, 1].min()
    xmax, ymax = v[:, 0].max(), v[:, 1].max()
    ext = max(xmax - xmin, ymax - ymin)

    # silueta del modelo en una grilla normalizada
    g = grid / ext * 0.9
    xy = np.c_[(v[:, 0] - xmin) * g + grid * 0.05, (ymax - v[:, 1]) * g + grid * 0.05]
    sil = np.isfinite(_zbuffer(xy, v[:, 2], mesh.faces, grid, grid))

    bx0, by0, bx1, by1 = _bbox(photo_mask)
    s0 = min((bx1 - bx0) / (xmax - xmin), (by1 - by0) / (ymax - ymin))
    ph, pw = photo_mask.shape
    gy, gx = np.where(np.ones((grid, grid), bool))
    # coordenadas de modelo de cada celda de la grilla
    mx = (gx + 0.5 - grid * 0.05) / g + xmin
    my = ymax - (gy + 0.5 - grid * 0.05) / g

    best = (-1, None)
    for ds in np.linspace(0.90, 1.10, 9):
        for asp in (0.94, 0.97, 1.0, 1.03, 1.06):
            sx, sy = s0 * ds, s0 * ds * asp
            tx0 = (bx0 + bx1) / 2 - sx * (xmin + xmax) / 2
            ty0 = (by0 + by1) / 2 + sy * (ymin + ymax) / 2
            for dx in np.linspace(-0.04, 0.04, 5) * (bx1 - bx0):
                for dy in np.linspace(-0.04, 0.04, 5) * (by1 - by0):
                    u = np.clip((mx * sx + tx0 + dx).astype(int), 0, pw - 1)
                    w = np.clip((-my * sy + ty0 + dy).astype(int), 0, ph - 1)
                    pm = photo_mask[w, u].reshape(grid, grid)
                    inter = (pm & sil).sum()
                    iou = inter / max((pm | sil).sum(), 1)
                    if iou > best[0]:
                        best = (iou, (sx, sy, tx0 + dx, ty0 + dy))
    return best[1], best[0]


def _choose_mirror(mesh, tex, photo_rgb, photo_mask, align):
    """Elige si la foto está espejada respecto al modelo (correlación de luminancia)."""
    sx, sy, tx, ty = align
    c = mesh.triangles_center
    front = mesh.face_normals[:, 2] > 0.5
    uv = mesh.visual.uv[mesh.faces].mean(1)
    th, tw = tex.shape[:2]
    tl = tex[np.clip(((1 - uv[:, 1]) * (th - 1)).astype(int), 0, th - 1),
             np.clip((uv[:, 0] * (tw - 1)).astype(int), 0, tw - 1)] @ [0.299, 0.587, 0.114]
    ph, pw = photo_mask.shape
    scores = {}
    for mirror in (False, True):
        cx = -c[:, 0] if mirror else c[:, 0]
        # con espejo hay que recentrar: el centro del modelo no cambia en la foto
        x = cx * sx + tx + (2 * c[:, 0].mean() * sx if mirror else 0)
        u = np.clip(x.astype(int), 0, pw - 1)
        w = np.clip((-c[:, 1] * sy + ty).astype(int), 0, ph - 1)
        ok = front & photo_mask[w, u]
        if ok.sum() < 20:
            scores[mirror] = -1
            continue
        pl = photo_rgb[w[ok], u[ok]] @ [0.299, 0.587, 0.114]
        scores[mirror] = np.corrcoef(pl, tl[ok])[0, 1]
    return scores[True] > scores[False] + 0.05


# ---------- proyección ----------
def project_photo(mesh, photo_rgba, min_facing=0.25, full_facing=0.65, erode_px=None):
    """
    Pinta la foto sobre la textura base del modelo (modifica mesh.visual.material).
    photo_rgba: PIL.Image RGBA con el fondo ya quitado (alpha = máscara).
    Devuelve un dict con métricas para el informe.
    """
    mat = mesh.visual.material
    tex_img = mat.baseColorTexture.convert("RGB")
    tex = np.asarray(tex_img).astype(np.float32) / 255
    th, tw = tex.shape[:2]

    photo = np.asarray(photo_rgba.convert("RGBA")).astype(np.float32) / 255
    photo_rgb, photo_a = photo[..., :3], photo[..., 3]
    photo_mask = photo_a > 0.5
    ph, pw = photo_mask.shape

    align, iou = _fit_alignment(mesh, photo_mask)
    sx, sy, tx, ty = align

    # contraer la máscara: los bordes mezclan objeto y fondo
    from scipy import ndimage
    if erode_px is None:
        erode_px = max(2, int(0.006 * max(ph, pw)))
    safe = ndimage.binary_erosion(photo_mask, iterations=erode_px)
    photo_a = photo_a * safe
    mirror = _choose_mirror(mesh, tex, photo_rgb, photo_mask, align)

    v = mesh.vertices
    xm = (-v[:, 0] + 2 * mesh.triangles_center[:, 0].mean()) if mirror else v[:, 0]
    pxy = np.c_[xm * sx + tx, -v[:, 1] * sy + ty]
    zbuf = _zbuffer(pxy, v[:, 2], mesh.faces, ph, pw)
    eps = 0.01 * mesh.extents.max()

    uvpx = np.c_[mesh.visual.uv[:, 0] * tw, (1 - mesh.visual.uv[:, 1]) * th]
    nz = mesh.face_normals[:, 2]
    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw), np.float32)
    painted = 0
    for fi in np.where(nz > min_facing)[0]:
        f = mesh.faces[fi]
        r = _raster_triangle(uvpx[f], th, tw)
        if r is None:
            continue
        xs, ys, bary = r
        p = bary @ pxy[f]
        z = bary @ v[f, 2]
        u = np.clip(p[:, 0].astype(int), 0, pw - 1)
        w_ = np.clip(p[:, 1].astype(int), 0, ph - 1)
        visible = z >= zbuf[w_, u] - eps
        wt = np.clip((nz[fi] - min_facing) / (full_facing - min_facing), 0, 1)
        wt = wt * photo_a[w_, u] * visible
        if not wt.any():
            continue
        np.add.at(acc, (ys, xs), photo_rgb[w_, u] * wt[:, None])
        np.add.at(wsum, (ys, xs), wt)
        painted += 1

    m = wsum > 0
    blend = np.clip(wsum, 0, 1)[..., None]
    out = tex.copy()
    out[m] = (acc[m] / wsum[m][:, None]) * blend[m] + tex[m] * (1 - blend[m])
    mat.baseColorTexture = Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))
    return {"projection_iou": round(float(iou), 3), "projection_mirror": bool(mirror),
            "projection_faces": int(painted)}


# ---------- proyección mejorada solo sobre el marco (lentes) ----------
def project_photo_frame(mesh, photo_rgba, face_mask, photo_lens_masks, align,
                        min_facing=0.30, full_facing=0.70, feather_px=4, erode_px=2):
    """
    Pega la foto real sobre el FRENTE DEL MARCO.
      - Usa la misma alineación foto-modelo que detecta los cristales (align).
      - Solo caras del marco (face_mask) que miran a la cámara y no están tapadas (z-buffer).
      - Solo toma píxeles del marco en la foto: nunca de los cristales (muestran la mesa)
        ni del fondo.
      - Mezcla suave según cuánto mira la cara al frente y cerca del borde del marco,
        para que no se note el corte con la textura de SF3D.
    """
    from scipy import ndimage

    mat = mesh.visual.material
    tex = np.asarray(mat.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    th, tw = tex.shape[:2]

    photo = np.asarray(photo_rgba.convert("RGBA")).astype(np.float32) / 255
    rgb = photo[..., :3]
    obj = photo[..., 3] > 0.5
    lens_px = np.zeros_like(obj)
    for l in photo_lens_masks or []:
        lens_px |= l
    lens_px = ndimage.binary_dilation(lens_px, iterations=3)
    frame_px = ndimage.binary_erosion(obj & ~lens_px, iterations=erode_px)
    if frame_px.sum() < 500:
        return {"projected": False}
    # peso por cercanía al borde del marco: 0 en el borde, 1 a feather_px píxeles adentro
    edge_w = np.clip(ndimage.distance_transform_edt(frame_px) / feather_px, 0, 1)
    ph, pw = frame_px.shape

    sx, sy, tx, ty = align
    v = mesh.vertices
    pxy = np.c_[v[:, 0] * sx + tx, -v[:, 1] * sy + ty]
    zbuf = _zbuffer(pxy, v[:, 2], mesh.faces, ph, pw)
    eps = 0.01 * mesh.extents.max()

    uvpx = np.c_[mesh.visual.uv[:, 0] * tw, (1 - mesh.visual.uv[:, 1]) * th]
    nz = mesh.face_normals[:, 2]
    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw), np.float32)
    painted = 0
    for fi in np.where(face_mask & (nz > min_facing))[0]:
        f = mesh.faces[fi]
        r = _raster_triangle(uvpx[f], th, tw)
        if r is None:
            continue
        xs, ys, bary = r
        p = bary @ pxy[f]
        z = bary @ v[f, 2]
        u = np.clip(p[:, 0].astype(int), 0, pw - 1)
        w_ = np.clip(p[:, 1].astype(int), 0, ph - 1)
        visible = z >= zbuf[w_, u] - eps
        facing = np.clip((nz[fi] - min_facing) / (full_facing - min_facing), 0, 1)
        wt = facing * edge_w[w_, u] * frame_px[w_, u] * visible
        if not wt.any():
            continue
        np.add.at(acc, (ys, xs), rgb[w_, u] * wt[:, None])
        np.add.at(wsum, (ys, xs), wt)
        painted += 1

    m = wsum > 1e-6
    blend = np.clip(wsum, 0, 1)[..., None]
    out = tex.copy()
    out[m] = (acc[m] / wsum[m][:, None]) * blend[m] + tex[m] * (1 - blend[m])
    mat.baseColorTexture = Image.fromarray((out * 255).clip(0, 255).astype(np.uint8))
    return {"projected": True, "projected_faces": int(painted),
            "projected_texels": int(m.sum())}
