import math
from app import features

TEXTO = "El gato come pescado. El gato come carne fresca."


def test_palabras_distintas_calculadas_a_mano():
    # 9 palabras, 6 distintas (el, gato, come, pescado, carne, fresca)
    assert math.isclose(features.caracteristicas_estilo(TEXTO)["palabras_distintas"], 6 / 9)


def test_longitud_de_oraciones_a_mano():
    car = features.caracteristicas_estilo(TEXTO)  # oraciones de 4 y 5 palabras
    assert math.isclose(car["long_oracion_media"], 4.5)
    assert math.isclose(car["long_oracion_desv"], 0.5)


def test_repeticion_de_trigramas_a_mano():
    # 7 grupos de tres palabras; "el gato come" aparece 2 veces -> 2/7
    assert math.isclose(features.caracteristicas_estilo(TEXTO)["repeticion_trigramas"], 2 / 7)


def test_descarta_fragmentos_de_menos_de_tres_palabras():
    assert features.dividir_oraciones("Hola amigo. Esta oración sí tiene suficientes palabras.") == [
        "Esta oración sí tiene suficientes palabras."]


def test_separa_por_saltos_de_linea():
    assert len(features.dividir_oraciones("Una línea con varias palabras\nOtra línea con varias palabras")) == 2


def test_vector_con_las_seis_caracteristicas_en_orden():
    car = features.extraer(TEXTO, [10.0, 20.0])
    assert list(car) == features.FEATURE_NAMES
    assert math.isclose(car["perplejidad_media"], 15.0)
    assert math.isclose(car["perplejidad_desv"], 5.0)


def test_sin_modelo_de_lenguaje_la_perplejidad_queda_vacia():
    car = features.extraer(TEXTO, None)
    assert car["perplejidad_media"] is None and car["perplejidad_desv"] is None
