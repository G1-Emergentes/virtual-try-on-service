"""
Miniatura de la prenda para las tarjetas de la app: la foto sin fondo.

PNG cuadrado con fondo transparente, el accesorio centrado y con margen. La app le pone
el color de fondo de la tarjeta. Se crea con el recorte que el motor ya hace antes de
reconstruir, así que no cuesta GPU extra.
"""
import io

import numpy as np
from PIL import Image

THUMBNAIL_SIZE = 512
MARGIN = 0.08          # 8 % de aire alrededor del accesorio


def make_thumbnail(cutout: Image.Image, size: int = THUMBNAIL_SIZE) -> bytes:
    rgba = cutout.convert("RGBA")
    alpha = np.asarray(rgba)[..., 3]
    ys, xs = np.where(alpha > 16)
    if len(xs) == 0:                                  # recorte vacío: imagen completa
        box = (0, 0, rgba.width, rgba.height)
    else:
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    item = rgba.crop(box)
    inner = int(size * (1 - 2 * MARGIN))
    item.thumbnail((inner, inner), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(item, ((size - item.width) // 2, (size - item.height) // 2), item)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
