"""Textos y utilidades de prueba. Los datos son de ejemplo; no hacen parte del corpus."""
import numpy as np

ORACIONES = [
    "La educación a distancia permite que muchas personas estudien sin desplazarse hasta una sede.",
    "Sin embargo, también exige disciplina y organización del tiempo por parte de los estudiantes.",
    "Las plataformas virtuales facilitan el acceso a los materiales de estudio durante todo el día.",
    "Por otra parte, la comunicación con los docentes depende de la conexión a internet de cada persona.",
    "En consecuencia, las universidades deben acompañar a sus estudiantes con tutorías claras y oportunas.",
    "El uso de herramientas digitales cambia la forma en que se elaboran y se entregan los trabajos escritos.",
    "Por eso es importante que los docentes cuenten con apoyos que les ayuden a revisar los documentos con cuidado.",
    "Finalmente, la integridad académica depende tanto de los estudiantes como de las reglas de la institución.",
]
TEXTO_VALIDO = " ".join(ORACIONES * 2)  # más de 100 palabras, en español
TEXTO_CORTO = "Este texto es demasiado corto para analizarlo."
TEXTO_LARGO = " ".join(ORACIONES * 20)  # más de 1.000 palabras
TEXTO_INGLES = ("Distance education allows many people to study without travelling to a campus, "
                "but it also requires discipline and good time management from every student. ") * 8


class LMFalso:
    """Modelo de lenguaje simulado: perplejidades fijas, sin descargar nada."""
    def perplejidades(self, oraciones):
        return [30.0 + 5 * i for i, _ in enumerate(oraciones)]


def modelo_sintetico(con_lm=True):
    """Modelo entrenado con datos sintéticos (verifica el código, no el desempeño)."""
    from app import features, modeling
    from scripts.crear_modelo_demo import datos_sinteticos
    nombres = list(features.FEATURE_NAMES if con_lm else features.FEATURES_ESTILO)
    X, y, g = datos_sinteticos(100)
    X = X if con_lm else X[:, 2:]
    bajo, alto = modeling.calibrar_umbrales(y, modeling.probabilidades_cv(X, y, g, nombres))
    return {"pipeline": modeling.entrenar_regresion(X, y, nombres), "features": nombres,
            "umbral_bajo": bajo, "umbral_alto": alto, "demo": False}
