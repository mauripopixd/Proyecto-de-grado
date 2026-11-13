"""Crea un MODELO DE DEMOSTRACIÓN con características sintéticas (no son textos reales).
Sirve solo para mostrar que el servicio funciona de extremo a extremo mientras no exista el corpus real.
Uso: python -m scripts.crear_modelo_demo [--sin-lm]"""
import argparse
import os

import joblib
import numpy as np

from app import features, modeling


def datos_sinteticos(n=200, semilla=modeling.SEMILLA):
    rng = np.random.default_rng(semilla)
    def clase(perp, desv, ttr, lon, lond, rep, n):
        return np.column_stack([np.clip(rng.normal(perp, perp * .25, n), 5, None), np.clip(rng.normal(desv, desv * .3, n), 1, None),
                                np.clip(rng.normal(ttr, .05, n), .2, 1), np.clip(rng.normal(lon, 3, n), 5, None),
                                np.clip(rng.normal(lond, 2, n), .5, None), np.clip(rng.normal(rep, .015, n), 0, None)])
    humano = clase(60, 40, .62, 21, 9, .02, n // 2)
    ia = clase(25, 12, .55, 24, 4.5, .05, n // 2)
    X = np.vstack([humano, ia]); y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y, np.arange(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="models/modelo_demo.joblib")
    ap.add_argument("--sin-lm", action="store_true")
    args = ap.parse_args()
    nombres = list(features.FEATURES_ESTILO if args.sin_lm else features.FEATURE_NAMES)
    X, y, grupos = datos_sinteticos()
    if args.sin_lm:
        X = X[:, 2:]
    bajo, alto = modeling.calibrar_umbrales(y, modeling.probabilidades_cv(X, y, grupos, nombres))
    pipe = modeling.entrenar_regresion(X, y, nombres)
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": nombres, "umbral_bajo": bajo, "umbral_alto": alto, "demo": True}, args.salida)
    print("Modelo de demostración (datos sintéticos) guardado en", args.salida)


if __name__ == "__main__":
    main()
