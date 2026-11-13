import math
import numpy as np
import pandas as pd
from app import features, modeling
from scripts import entrenar as entrenar_mod
from scripts.crear_modelo_demo import datos_sinteticos


def test_division_por_grupos_no_mezcla_grupos():
    grupos = np.repeat(np.arange(30), 2)  # 2 textos por autor/grupo
    tr, te = modeling.dividir_por_grupos(grupos, 0.2)
    assert not set(grupos[tr]) & set(grupos[te])
    assert len(te) > 0 and len(tr) + len(te) == len(grupos)


def test_division_reproducible_con_semilla_fija():
    grupos = np.repeat(np.arange(30), 2)
    a = modeling.dividir_por_grupos(grupos, 0.2, semilla=7)
    b = modeling.dividir_por_grupos(grupos, 0.2, semilla=7)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_umbral_alto_limita_los_falsos_positivos():
    y = np.array([0] * 10 + [1] * 10)
    p = np.array([.05, .1, .15, .2, .25, .3, .35, .4, .45, .9] + [.6, .65, .7, .75, .8, .85, .9, .92, .95, .97])
    bajo, alto = modeling.calibrar_umbrales(y, p, fpr_max=0.1)
    assert np.mean(p[y == 0] >= alto) <= 0.1
    assert bajo <= alto


def test_etiquetas_segun_umbrales():
    assert modeling.etiquetar(0.1, 0.3, 0.8) == "baja_probabilidad_ia"
    assert modeling.etiquetar(0.5, 0.3, 0.8) == "indeterminado"
    assert modeling.etiquetar(0.9, 0.3, 0.8) == "alta_probabilidad_ia"


def test_metricas_calculadas_a_mano():
    m = modeling.metricas([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0])  # vp=2 fn=1 fp=1 vn=2
    assert m["matriz"] == {"vn": 2, "fp": 1, "fn": 1, "vp": 2}
    assert math.isclose(m["precision"], 2 / 3) and math.isclose(m["recall"], 2 / 3)
    assert math.isclose(m["tasa_falsos_positivos"], 1 / 3)


def test_linea_base_separa_por_perplejidad():
    x, y = [10, 12, 14, 50, 60, 70], [1, 1, 1, 0, 0, 0]
    u = modeling.ajustar_linea_base(x, y)
    assert (modeling.predecir_linea_base(x, u) == np.array(y)).all()


def test_entrenamiento_reproducible_y_genera_informe(tmp_path):
    X, y, g = datos_sinteticos(100)
    df = pd.DataFrame(X, columns=features.FEATURE_NAMES)
    df["clase"] = np.where(y == 1, "ia", "humano")
    df["grupo"] = g // 2  # dos textos por autor
    r1 = entrenar_mod.entrenar(df, features.FEATURE_NAMES)
    r2 = entrenar_mod.entrenar(df, features.FEATURE_NAMES)
    assert r1["resultados"] == r2["resultados"]
    assert r1["resultados"]["n_prueba"] > 0
    ruta = tmp_path / "informe.md"
    entrenar_mod.escribir_informe(r1["resultados"], str(ruta))
    assert "Matriz de confusión" in ruta.read_text(encoding="utf8")
