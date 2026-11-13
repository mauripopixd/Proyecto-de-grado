"""Entrena el clasificador, lo evalúa en un conjunto de prueba separado por grupos y escribe el informe.
Uso: python -m scripts.entrenar [--caracteristicas data/caracteristicas.csv] [--modelo models/modelo.joblib]"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from app import features, modeling


def entrenar(df, nombres, prueba=0.2, semilla=modeling.SEMILLA):
    nombres = list(nombres)
    y = (df["clase"] == "ia").astype(int).values
    X = df[nombres].values.astype(float)
    g = df["grupo"].values
    if min(int(y.sum()), int((1 - y).sum())) < 6:
        raise ValueError("Hay muy pocos textos en una de las clases (mínimo 6 por clase).")
    tr, te = modeling.dividir_por_grupos(g, prueba, semilla)
    if len(set(y[te].tolist())) < 2:
        raise ValueError("El conjunto de prueba quedó con una sola clase; ajuste la proporción o agregue textos.")
    proba_cv = modeling.probabilidades_cv(X[tr], y[tr], g[tr], nombres, semilla)
    bajo, alto = modeling.calibrar_umbrales(y[tr], proba_cv)
    pipe = modeling.entrenar_regresion(X[tr], y[tr], nombres, semilla)
    proba = pipe.predict_proba(X[te])[:, 1]
    res = {"n_entrenamiento": int(len(tr)), "n_prueba": int(len(te)), "caracteristicas": nombres,
           "umbral_bajo": bajo, "umbral_alto": alto,
           "clasificador": modeling.metricas(y[te], (proba >= 0.5).astype(int)),
           "validacion_cruzada": modeling.metricas(y[tr], (proba_cv >= 0.5).astype(int)),
           "indeterminados_en_prueba": int(sum(modeling.etiquetar(p, bajo, alto) == "indeterminado" for p in proba)),
           "coeficientes": {n: float(c) for n, c in zip(nombres, pipe[-1].coef_[0])}}
    if "perplejidad_media" in nombres:
        i = nombres.index("perplejidad_media")
        u = modeling.ajustar_linea_base(X[tr][:, i], y[tr])
        res["linea_base"] = modeling.metricas(y[te], modeling.predecir_linea_base(X[te][:, i], u))
    bundle = {"pipeline": pipe, "features": nombres, "umbral_bajo": bajo, "umbral_alto": alto, "demo": False}
    return {"resultados": res, "modelo": bundle}


def _fila(nombre, m):
    return f"| {nombre} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['tasa_falsos_positivos']:.2f} |"


def escribir_informe(r, ruta):
    t = ["# Informe de evaluación", "",
         f"Textos de entrenamiento: {r['n_entrenamiento']}. Textos de prueba: {r['n_prueba']}. Con tan pocos textos, "
         "cada error mueve mucho las métricas; los resultados son orientativos.", "",
         "## Conjunto de prueba (clase positiva: IA)", "",
         "| Modelo | Precisión | Recall | F1 | Tasa de falsos positivos |", "| --- | --- | --- | --- | --- |"]
    if "linea_base" in r:
        t.append(_fila("Línea base (perplejidad)", r["linea_base"]))
    t.append(_fila("Clasificador", r["clasificador"]))
    m = r["clasificador"]["matriz"]
    t += ["", "## Matriz de confusión del clasificador en el conjunto de prueba (umbral 0,5)", "",
          "| | Predicho humano | Predicho IA |", "| --- | --- | --- |",
          f"| Real humano | {m['vn']} | {m['fp']} |", f"| Real IA | {m['fn']} | {m['vp']} |", "",
          f"Textos de prueba que quedarían como «indeterminado» con los umbrales calibrados: {r['indeterminados_en_prueba']}.", "",
          "## Validación cruzada por grupos (entrenamiento)", "",
          "| Precisión | Recall | F1 | Tasa de falsos positivos |", "| --- | --- | --- | --- |",
          "| {precision:.2f} | {recall:.2f} | {f1:.2f} | {tasa_falsos_positivos:.2f} |".format(**r["validacion_cruzada"]), "",
          "## Coeficientes de la regresión logística (variables estandarizadas)", ""]
    t += [f"- {n}: {c:.3f}" for n, c in r["coeficientes"].items()]
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(t) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caracteristicas", default="data/caracteristicas.csv")
    ap.add_argument("--modelo", default="models/modelo.joblib")
    ap.add_argument("--reportes", default="reports")
    ap.add_argument("--prueba", type=float, default=0.2)
    a = ap.parse_args()
    df = pd.read_csv(a.caracteristicas)
    nombres = [n for n in features.FEATURE_NAMES if n in df.columns and df[n].notna().all()]
    try:
        r = entrenar(df, nombres, a.prueba)
    except ValueError as e:
        raise SystemExit(str(e))
    os.makedirs(os.path.dirname(a.modelo) or ".", exist_ok=True)
    joblib.dump(r["modelo"], a.modelo)
    os.makedirs(a.reportes, exist_ok=True)
    with open(os.path.join(a.reportes, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(r["resultados"], f, indent=2, ensure_ascii=False)
    escribir_informe(r["resultados"], os.path.join(a.reportes, "informe.md"))
    print(open(os.path.join(a.reportes, "informe.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
