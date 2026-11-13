import pytest
from fastapi.testclient import TestClient
from app.main import crear_app
from tests.datos import LMFalso, TEXTO_CORTO, TEXTO_INGLES, TEXTO_VALIDO, modelo_sintetico


@pytest.fixture
def cliente():
    return TestClient(crear_app(modelo=modelo_sintetico(), lm=LMFalso()))


def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200 and r.json()["modelo_cargado"] is True


def test_clasificar_texto_valido(cliente):
    r = cliente.post("/clasificar", json={"texto": TEXTO_VALIDO})
    assert r.status_code == 200
    cuerpo = r.json()
    assert {"puntaje_ia", "etiqueta", "num_palabras", "caracteristicas", "tiempo_ms", "advertencia"} <= set(cuerpo)


def test_texto_corto_da_422(cliente):
    r = cliente.post("/clasificar", json={"texto": TEXTO_CORTO})
    assert r.status_code == 422 and r.json()["error"]["codigo"] == "texto_corto"


def test_texto_en_ingles_da_422(cliente):
    r = cliente.post("/clasificar", json={"texto": TEXTO_INGLES})
    assert r.status_code == 422 and r.json()["error"]["codigo"] == "idioma_no_espanol"


def test_solicitud_sin_texto_da_422(cliente):
    assert cliente.post("/clasificar", json={}).status_code == 422


def test_sin_modelo_da_503(tmp_path):
    c = TestClient(crear_app(modelo_path=str(tmp_path / "no_existe.joblib")))
    r = c.post("/clasificar", json={"texto": TEXTO_VALIDO})
    assert r.status_code == 503 and r.json()["error"]["codigo"] == "modelo_no_disponible"
    assert c.get("/salud").json()["modelo_cargado"] is False


def test_la_consulta_no_crea_archivos(cliente, tmp_path, monkeypatch):  # RNF2
    monkeypatch.chdir(tmp_path)
    cliente.post("/clasificar", json={"texto": TEXTO_VALIDO})
    assert list(tmp_path.iterdir()) == []
