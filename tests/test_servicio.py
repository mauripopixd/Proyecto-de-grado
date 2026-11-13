import os
import tempfile
import pytest
from app import servicio
from tests.datos import (LMFalso, TEXTO_CORTO, TEXTO_INGLES, TEXTO_LARGO, TEXTO_VALIDO, modelo_sintetico)


def codigo_de_error(texto, modelo, lm=None):
    with pytest.raises(servicio.ErrorApi) as e:
        servicio.clasificar(texto, modelo, lm)
    return e.value.codigo, e.value.http


def test_texto_valido_devuelve_puntaje_etiqueta_y_advertencia():
    r = servicio.clasificar(TEXTO_VALIDO, modelo_sintetico(), LMFalso())
    assert 0 <= r["puntaje_ia"] <= 1
    assert r["etiqueta"] in {"baja_probabilidad_ia", "indeterminado", "alta_probabilidad_ia"}
    assert r["num_palabras"] >= 100 and "orientativo" in r["advertencia"]
    assert len(r["caracteristicas"]) == 6 and "tiempo_ms" in r


def test_texto_corto():
    assert codigo_de_error(TEXTO_CORTO, modelo_sintetico(), LMFalso()) == ("texto_corto", 422)


def test_texto_largo():
    assert codigo_de_error(TEXTO_LARGO, modelo_sintetico(), LMFalso()) == ("texto_largo", 422)


def test_texto_en_ingles():
    assert codigo_de_error(TEXTO_INGLES, modelo_sintetico(), LMFalso()) == ("idioma_no_espanol", 422)


def test_sin_modelo_entrenado():
    assert codigo_de_error(TEXTO_VALIDO, None) == ("modelo_no_disponible", 503)


def test_modelo_solo_estilo_no_necesita_modelo_de_lenguaje():
    r = servicio.clasificar(TEXTO_VALIDO, modelo_sintetico(con_lm=False), None)
    assert r["caracteristicas"]["perplejidad_media"] is None


def test_modelo_de_demostracion_se_advierte():
    m = modelo_sintetico(con_lm=False); m["demo"] = True
    r = servicio.clasificar(TEXTO_VALIDO, m, None)
    assert r["modelo_demo"] is True and "DEMOSTRACIÓN" in r["advertencia"]


def test_no_guarda_archivos():  # RNF2
    with tempfile.TemporaryDirectory() as d:
        antes = os.getcwd(); os.chdir(d)
        try:
            servicio.clasificar(TEXTO_VALIDO, modelo_sintetico(), LMFalso())
            assert os.listdir(d) == []
        finally:
            os.chdir(antes)
