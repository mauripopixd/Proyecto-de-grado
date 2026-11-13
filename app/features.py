"""Extracción de las seis características de un texto (Tabla 4 del documento maestro)."""
import re
from collections import Counter

import numpy as np

FEATURE_NAMES = ["perplejidad_media", "perplejidad_desv", "palabras_distintas",
                 "long_oracion_media", "long_oracion_desv", "repeticion_trigramas"]
FEATURES_ESTILO = FEATURE_NAMES[2:]
_PALABRA = re.compile(r"\w+", re.UNICODE)


def palabras(texto):
    return [p.lower() for p in _PALABRA.findall(texto)]


def dividir_oraciones(texto, min_palabras=3):
    """Separa por signos de puntuación final y saltos de línea; descarta fragmentos de menos de tres palabras."""
    partes = re.split(r"(?<=[.!?…])\s+|\n+", texto)
    return [p.strip() for p in partes if len(palabras(p)) >= min_palabras]


def caracteristicas_estilo(texto):
    pal = palabras(texto)
    largos = [len(palabras(o)) for o in dividir_oraciones(texto)]
    if not pal or not largos:
        raise ValueError("texto sin oraciones válidas")
    trigramas = [tuple(pal[i:i + 3]) for i in range(len(pal) - 2)]
    cuenta = Counter(trigramas)
    rep = sum(1 for t in trigramas if cuenta[t] > 1) / len(trigramas) if trigramas else 0.0
    return {
        "palabras_distintas": len(set(pal)) / len(pal),
        "long_oracion_media": float(np.mean(largos)),
        "long_oracion_desv": float(np.std(largos)),
        "repeticion_trigramas": rep,
    }


def extraer(texto, perplejidades=None):
    """Seis características en orden. Sin perplejidades (modelo de lenguaje ausente) las dos primeras quedan en None."""
    est = caracteristicas_estilo(texto)
    if perplejidades:
        p = np.asarray(perplejidades, dtype=float)
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("perplejidad inválida")
        ppl = {"perplejidad_media": float(p.mean()), "perplejidad_desv": float(p.std())}
    else:
        ppl = {"perplejidad_media": None, "perplejidad_desv": None}
    return {**ppl, **est}
