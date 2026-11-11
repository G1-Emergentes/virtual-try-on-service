"""
Configuración de pytest.

Las pruebas no usan la GPU ni el modelo de IA: el motor 3D se reemplaza por dobles.
Si torch no está instalado (por ejemplo, fuera del contenedor), se registra un módulo
mínimo para poder importar los routers.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    fake = types.ModuleType("torch")

    class _OutOfMemoryError(RuntimeError):
        pass

    fake.cuda = types.SimpleNamespace(is_available=lambda: False, OutOfMemoryError=_OutOfMemoryError,
                                      empty_cache=lambda: None)
    sys.modules["torch"] = fake
