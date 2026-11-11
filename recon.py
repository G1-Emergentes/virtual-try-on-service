"""
Modelo de reconstrucción imagen -> 3D, intercambiable.

  RECON_MODEL=spar3d  (por defecto) Stable Point Aware 3D: sucesor de SF3D, predice mejor
                      la parte de atrás con una nube de puntos. ~7 GB de VRAM en modo
                      de bajo consumo (SPAR3D_LOW_VRAM=1).
  RECON_MODEL=sf3d    Stable Fast 3D: ~6 GB de VRAM. Respaldo si SPAR3D no entra en la GPU.

Las dos usan la misma interfaz: cutout() quita el fondo, prepare() recorta y centra
el objeto, run() genera la malla con textura.

El Dockerfile instala solo uno de los dos (comparten nombres de paquetes internos),
así que este valor debe coincidir con el usado al construir la imagen.
"""
import os

RECON_MODEL = os.getenv("RECON_MODEL", "spar3d").lower()


class Reconstructor:
    def __init__(self, device: str):
        self.device = device
        self.name = RECON_MODEL
        if self.name == "spar3d":
            from transparent_background import Remover

            from spar3d.system import SPAR3D

            low_vram = os.getenv("SPAR3D_LOW_VRAM", "1") == "1"
            self.model = SPAR3D.from_pretrained(
                "stabilityai/stable-point-aware-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
                low_vram_mode=low_vram,
            )
            self.remover = Remover(device=device)
            self.foreground = float(os.getenv("FOREGROUND_RATIO", "1.3"))
            self.low_vram = low_vram
        elif self.name == "sf3d":
            import rembg

            from sf3d.system import SF3D

            self.model = SF3D.from_pretrained(
                "stabilityai/stable-fast-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            self.remover = rembg.new_session(os.getenv("REMBG_MODEL", "u2net"))
            self.foreground = float(os.getenv("FOREGROUND_RATIO", "0.85"))
            self.low_vram = False
        else:
            raise ValueError(f"RECON_MODEL no soportado: {self.name} (usa spar3d o sf3d)")
        self.model.to(device)
        self.model.eval()

    def cutout(self, image):
        """Imagen RGBA con el fondo quitado (alpha = máscara del objeto)."""
        if self.name == "spar3d":
            from spar3d.utils import remove_background
        else:
            from sf3d.utils import remove_background
        return remove_background(image.convert("RGBA"), self.remover)

    def prepare(self, cutout):
        """Recorta y centra el objeto como espera el modelo."""
        if self.name == "spar3d":
            from spar3d.utils import foreground_crop

            return foreground_crop(cutout, self.foreground)
        from sf3d.utils import resize_foreground

        return resize_foreground(cutout, self.foreground)

    def run(self, image, texture_res: int, remesh: str):
        mesh, _ = self.model.run_image(image, bake_resolution=texture_res, remesh=remesh)
        return mesh
