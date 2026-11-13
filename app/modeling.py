"""Línea base, regresión logística, división por grupos, umbrales y métricas."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEMILLA = 42


class LogPerplejidad(BaseEstimator, TransformerMixin):
    """Aplica logaritmo a la perplejidad media (columna indicada) para estabilizar su escala."""
    def __init__(self, indice=None):
        self.indice = indice

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        if self.indice is not None:
            X[:, self.indice] = np.log(X[:, self.indice])
        return X


def _crear(nombres, semilla=SEMILLA):
    i = nombres.index("perplejidad_media") if "perplejidad_media" in nombres else None
    return make_pipeline(LogPerplejidad(i), StandardScaler(), LogisticRegression(max_iter=1000, random_state=semilla))


def entrenar_regresion(X, y, nombres, semilla=SEMILLA):
    return _crear(list(nombres), semilla).fit(np.asarray(X, float), np.asarray(y))


def probabilidades_cv(X, y, grupos, nombres, semilla=SEMILLA):
    """Probabilidades de IA obtenidas con validación cruzada por grupos (ningún grupo en entrenamiento y prueba a la vez)."""
    k = min(5, len(set(np.asarray(grupos).tolist())))
    if k < 2:
        raise ValueError("se necesitan al menos 2 grupos para validar")
    return cross_val_predict(_crear(list(nombres), semilla), np.asarray(X, float), np.asarray(y), groups=np.asarray(grupos),
                             cv=GroupKFold(n_splits=k), method="predict_proba")[:, 1]


def dividir_por_grupos(grupos, prueba=0.2, semilla=SEMILLA):
    grupos = np.asarray(grupos)
    return next(GroupShuffleSplit(n_splits=1, test_size=prueba, random_state=semilla).split(np.zeros(len(grupos)), groups=grupos))


def calibrar_umbrales(y, proba, fpr_max=0.05, fnr_max=0.05):
    """'alto': el menor umbral con a lo sumo fpr_max de falsos positivos entre los humanos.
    'bajo': el mayor umbral que deja por debajo a lo sumo fnr_max de los textos de IA."""
    y, proba = np.asarray(y), np.asarray(proba, float)
    h, a = proba[y == 0], proba[y == 1]
    alto = 1.0
    for t in sorted(set(proba.tolist()) | {float(proba.max()) + 1e-9}):
        if len(h) == 0 or (h >= t).mean() <= fpr_max:
            alto = float(t)
            break
    bajo = 0.0
    for t in sorted(set(proba.tolist()), reverse=True):
        if len(a) == 0 or (a < t).mean() <= fnr_max:
            bajo = float(t)
            break
    return min(bajo, alto), alto


def etiquetar(p, bajo, alto):
    if p >= alto:
        return "alta_probabilidad_ia"
    if p < bajo:
        return "baja_probabilidad_ia"
    return "indeterminado"


def metricas(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    vn, fp, fn, vp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {"precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "tasa_falsos_positivos": fp / (fp + vn) if (fp + vn) else 0.0,
            "matriz": {"vn": vn, "fp": fp, "fn": fn, "vp": vp}}


def ajustar_linea_base(ppl, y):
    """Línea base: es IA si la perplejidad media es menor que el umbral que maximiza F1 en entrenamiento."""
    ppl, y = np.asarray(ppl, float), np.asarray(y)
    mejor, mejor_f1 = None, -1.0
    for t in np.unique(ppl):
        f = f1_score(y, (ppl <= t).astype(int), zero_division=0)
        if f > mejor_f1:
            mejor, mejor_f1 = float(t) + 1e-9, f
    return mejor


def predecir_linea_base(ppl, umbral):
    return (np.asarray(ppl, float) < umbral).astype(int)
