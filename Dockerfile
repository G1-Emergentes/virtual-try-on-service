# Imagen base: PyTorch + CUDA 12.4 + compilador nvcc ("devel")
# nvcc hace falta para compilar texture_baker y uv_unwrapper (SPAR3D y SF3D los traen)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Modelo de reconstrucción: spar3d (por defecto) o sf3d.
# Solo se instala uno: los dos traen paquetes internos con el mismo nombre.
ARG RECON_MODEL=spar3d

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # Durante el build no hay GPU: forzamos compilar con CUDA
    USE_CUDA=1 \
    # 8.9 = arquitectura Ada Lovelace (RTX 4060)
    TORCH_CUDA_ARCH_LIST="8.9" \
    USE_NATIVE_ARCH=0 \
    HF_HOME=/cache/huggingface \
    U2NET_HOME=/cache/u2net \
    TRANSPARENT_BACKGROUND_FILE_PATH=/cache \
    RECON_MODEL=${RECON_MODEL}

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Congelar las versiones de la imagen base: sin esto, dependencias de SPAR3D sin versión
# fija (kornia, timm...) hacen que pip reemplace torch (CUDA 12.4 -> 13) y suba setuptools
# a una versión sin pkg_resources. PIP_CONSTRAINT aplica a TODOS los pip install siguientes,
# incluidos los entornos aislados de compilación.
RUN python -c "import importlib, importlib.util; [print(n + '==' + importlib.import_module(n).__version__.split('+')[0]) for n in ('torch','torchvision','torchaudio') if importlib.util.find_spec(n)]" > /opt/constraints.txt \
    && printf "setuptools==69.5.1\nnumpy==1.26.4\n" >> /opt/constraints.txt
RUN cat /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt

WORKDIR /opt
# Commits fijos para que el build sea reproducible
RUN if [ "$RECON_MODEL" = "spar3d" ]; then \
        git clone https://github.com/Stability-AI/stable-point-aware-3d.git recon \
        && cd recon && git checkout fdc311b16809e6a8adc2f5a3407ebb3db1a95bd1; \
    else \
        git clone https://github.com/Stability-AI/stable-fast-3d.git recon \
        && cd recon && git checkout ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2; \
    fi

WORKDIR /opt/recon
RUN pip install -U setuptools==69.5.1 wheel \
    && grep -v "^\./" requirements.txt | grep -v "^git+" > req-base.txt \
    && pip install -r req-base.txt \
    # Paquetes desde GitHub (CLIP, AlphaCLIP en SPAR3D): sin aislamiento, para que usen
    # setuptools 69.5.1 (las versiones nuevas ya no traen pkg_resources)
    && (grep "^git+" requirements.txt > req-git.txt || true) \
    && if [ -s req-git.txt ]; then pip install --no-build-isolation -r req-git.txt; fi \
    && if [ -f requirements-remesh.txt ]; then pip install -r requirements-remesh.txt; fi \
    # Extensiones C++/CUDA: necesitan ver el torch ya instalado
    && pip install --no-build-isolation ./texture_baker/ ./uv_unwrapper/

# Dependencias de la API
# flet 0.25.2: transparent-background (quitafondos de SPAR3D) importa ElevatedButton e icons,
# que flet 0.28+ y 1.x ya no tienen
RUN pip install fastapi==0.115.6 "uvicorn[standard]==0.32.1" python-multipart==0.0.20 \
    scipy==1.13.1 matplotlib==3.9.2 "google-genai>=1.20,<3" "httpx>=0.27,<1" \
    && if [ "$RECON_MODEL" = "spar3d" ]; then pip install "flet==0.25.2"; fi

ENV PYTHONPATH=/opt/recon
WORKDIR /app
# Motor 3D (módulos de la raíz) + servicio en capas (paquete tryon) + raíz de composición
COPY recon.py postprocess.py projection.py gemini_check.py material_presets.py ./
COPY tryon ./tryon
COPY app.py ./
# biblioteca de materiales: presets (JSON) + texturas de ambientCG (CC0)
COPY materials ./materials

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
