"""
Virtual Try-On Service: genera el modelo 3D (.glb) de un accesorio a partir de su foto.

Capas (arquitectura limpia + DDD):
- domain/          Reglas del negocio. No depende de FastAPI, de la GPU ni de HTTP.
- application/     Casos de uso: orquestan el dominio a través de puertos.
- infrastructure/  Adaptadores: motor 3D (SPAR3D), Digital Closet, disco.
- interfaces/      Entrada HTTP (FastAPI): traduce peticiones a casos de uso.

Las dependencias apuntan siempre hacia dentro: interfaces -> application -> domain.
La infraestructura implementa los puertos que define el dominio.
"""
