"""Convierte cada texto del corpus en su vector de características.
Uso: python -m scripts.extraer_caracteristicas [--corpus data/corpus] [--salida data/caracteristicas.csv] [--sin-lm]
El corpus es una carpeta con metadatos.csv (id,clase,autor,grupo,archivo) y los textos en textos/."""
import argparse
import csv
import os

from app import features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--salida", default="data/caracteristicas.csv")
    ap.add_argument("--sin-lm", action="store_true", help="solo características de estilo (sin perplejidad)")
    a = ap.parse_args()
    lm = None
    if not a.sin_lm:
        from app.perplexity import PerplejidadLM
        lm = PerplejidadLM()
    nombres = features.FEATURES_ESTILO if a.sin_lm else features.FEATURE_NAMES
    with open(os.path.join(a.corpus, "metadatos.csv"), encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    salida, omitidos = [], 0
    for r in filas:
        with open(os.path.join(a.corpus, "textos", r["archivo"]), encoding="utf-8") as f:
            texto = f.read()
        try:
            c = features.extraer(texto, lm.perplejidades(features.dividir_oraciones(texto)) if lm else None)
        except ValueError as e:
            print(f"Omitido {r['id']}: {e}")
            omitidos += 1
            continue
        salida.append({"id": r["id"], "clase": r["clase"], "autor": r["autor"],
                       "grupo": r.get("grupo") or r["autor"], **{n: c[n] for n in nombres}})
    os.makedirs(os.path.dirname(a.salida) or ".", exist_ok=True)
    with open(a.salida, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "clase", "autor", "grupo", *nombres])
        w.writeheader()
        w.writerows(salida)
    print(f"{len(salida)} textos procesados, {omitidos} omitidos -> {a.salida}")


if __name__ == "__main__":
    main()
