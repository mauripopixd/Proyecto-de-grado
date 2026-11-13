"""Lógica del servicio: valida, extrae características, puntúa y etiqueta. No guarda nada (RNF2)."""
import time

import numpy as np

from . import features, modeling

ADVERTENCIA = "Resultado orientativo; no prueba autoría."
ADVERTENCIA_DEMO = ("MODELO DE DEMOSTRACIÓN entrenado con datos sintéticos: solo muestra el funcionamiento del servicio; "
                    "no sirve para evaluar textos reales. " + ADVERTENCIA)
MIN_PALABRAS, MAX_PALABRAS = 100, 1000
_ES = set("de la que el en y a los del se las por un para con no una su al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también me hasta hay donde quien desde todo nos durante es son fue ser".split())
_EN = set("the of and to in is that it for as with was on are by this be or from an have not at which but their".split())


class ErrorApi(Exception):
    def __init__(self, codigo, mensaje, http=422):
        super().__init__(mensaje)
        self.codigo, self.mensaje, self.http = codigo, mensaje, http


def es_espanol(pal):
    if not pal:
        return False
    es = sum(p in _ES for p in pal) / len(pal)
    en = sum(p in _EN for p in pal) / len(pal)
    return es >= 0.15 and es > en


def clasificar(texto, modelo, lm=None):
    t0 = time.perf_counter()
    if modelo is None:
        raise ErrorApi("modelo_no_disponible", "El modelo aún no está entrenado.", 503)
    pal = features.palabras(texto)
    if len(pal) < MIN_PALABRAS:
        raise ErrorApi("texto_corto", f"El texto tiene menos de {MIN_PALABRAS} palabras.")
    if len(pal) > MAX_PALABRAS:
        raise ErrorApi("texto_largo", f"El texto tiene más de {MAX_PALABRAS} palabras.")
    if not es_espanol(pal):
        raise ErrorApi("idioma_no_espanol", "El texto no parece estar en español.")
    nombres = modelo["features"]
    ppl = None
    if "perplejidad_media" in nombres:
        if lm is None:
            raise ErrorApi("modelo_de_lenguaje_no_disponible", "Este modelo necesita el modelo de lenguaje y no está cargado.", 503)
        ppl = lm.perplejidades(features.dividir_oraciones(texto))
    try:
        car = features.extraer(texto, ppl)
    except ValueError as e:
        raise ErrorApi("caracteristicas_invalidas", str(e))
    x = np.array([[car[n] for n in nombres]], dtype=float)
    p = float(modelo["pipeline"].predict_proba(x)[0, 1])
    demo = bool(modelo.get("demo", False))
    return {"puntaje_ia": round(p, 4),
            "etiqueta": modeling.etiquetar(p, modelo["umbral_bajo"], modelo["umbral_alto"]),
            "num_palabras": len(pal),
            "caracteristicas": {k: (None if v is None else round(v, 4)) for k, v in car.items()},
            "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1),
            "modelo_demo": demo,
            "advertencia": ADVERTENCIA_DEMO if demo else ADVERTENCIA}


def clasificar_documento(texto, modelo, lm=None):
    """Analiza un documento completo: lo divide en bloques de ~500 palabras y promedia el puntaje."""
    from . import documentos
    t0 = time.perf_counter()
    if modelo is None:
        raise ErrorApi("modelo_no_disponible", "El modelo aún no está entrenado.", 503)
    pal = features.palabras(texto)
    if len(pal) < MIN_PALABRAS:
        raise ErrorApi("texto_corto", f"El documento tiene menos de {MIN_PALABRAS} palabras.")
    bloques, truncado = documentos.dividir_bloques(texto)   # el idioma se verifica bloque a bloque
    resultados, omitidos = [], 0
    for i, b in enumerate(bloques, 1):
        try:
            r = clasificar(b, modelo, lm)
        except ErrorApi as e:
            if e.codigo in ("idioma_no_espanol", "texto_corto", "texto_largo", "caracteristicas_invalidas"):
                omitidos += 1
                continue
            raise
        resultados.append({"bloque": i, "palabras": r["num_palabras"], "puntaje_ia": r["puntaje_ia"], "etiqueta": r["etiqueta"]})
    if not resultados:
        if omitidos and not es_espanol(pal):
            raise ErrorApi("idioma_no_espanol", "El documento no parece tener texto corrido en español (¿es una lista de referencias, una tabla, código o está en otro idioma?).")
        raise ErrorApi("sin_bloques_analizables", "No se pudo analizar ningún fragmento del documento: "
                       "verifique que tenga párrafos de texto en español (no solo tablas, código o referencias).")
    total = sum(r["palabras"] for r in resultados)
    media = sum(r["puntaje_ia"] * r["palabras"] for r in resultados) / total
    demo = bool(modelo.get("demo", False))
    return {"puntaje_ia": round(media, 4),
            "etiqueta": modeling.etiquetar(media, modelo["umbral_bajo"], modelo["umbral_alto"]),
            "num_palabras": len(pal), "palabras_analizadas": total,
            "bloques": resultados, "bloques_omitidos": omitidos, "truncado": truncado,
            "tiempo_ms": round((time.perf_counter() - t0) * 1000, 1),
            "modelo_demo": demo,
            "advertencia": ADVERTENCIA_DEMO if demo else ADVERTENCIA}
