import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from app import documentos
from app.main import crear_app
from app.servicio import ErrorApi
from tests.datos import LMFalso, TEXTO_INGLES, TEXTO_LARGO, TEXTO_VALIDO, modelo_sintetico


@pytest.fixture
def cliente():
    return TestClient(crear_app(modelo=modelo_sintetico(), lm=LMFalso()))


def _docx(texto):
    cuerpo = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in texto.split(". "))
    xml = f'<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{cuerpo}</w:body></w:document>'
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("word/document.xml", xml)
    return b.getvalue()


def _pdf(texto):
    flujo = f"BT /F1 10 Tf 40 700 Td ({texto}) Tj ET".encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(flujo) + flujo + b"\nendstream", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, pos = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        pos.append(len(out)); out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % p for p in pos)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, x)


def test_lee_txt_docx_y_pdf():
    assert documentos.leer_documento("a.txt", "Hola   mundo\nnuevo".encode("utf-8")) == "Hola mundo nuevo"
    assert "educación" in documentos.leer_documento("a.docx", _docx("La educación avanza. Otra frase aquí"))
    assert "Hola PDF" in documentos.leer_documento("a.pdf", _pdf("Hola PDF"))


def test_formato_no_soportado_y_vacio():
    with pytest.raises(ErrorApi) as e:
        documentos.leer_documento("a.exe", b"x")
    assert e.value.http == 415
    with pytest.raises(ErrorApi):
        documentos.leer_documento("a.txt", b"   ")


def test_dividir_bloques_une_el_resto_corto_y_limita():
    bloques, trunc = documentos.dividir_bloques(" ".join(["p"] * 1050))
    assert [len(b.split()) for b in bloques] == [500, 550] and not trunc
    bloques, trunc = documentos.dividir_bloques(" ".join(["p"] * 6000))
    assert len(bloques) == documentos.MAX_BLOQUES and trunc


def test_endpoint_archivo_txt_y_texto_largo(cliente):
    r = cliente.post("/clasificar-archivo", files={"archivo": ("t.txt", TEXTO_VALIDO.encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["bloques"][0]["palabras"] > 100
    r = cliente.post("/clasificar-archivo", files={"archivo": ("l.txt", TEXTO_LARGO.encode(), "text/plain")})
    cuerpo = r.json()
    assert r.status_code == 200 and len(cuerpo["bloques"]) >= 2 and 0 <= cuerpo["puntaje_ia"] <= 1


def test_endpoint_archivo_errores(cliente):
    r = cliente.post("/clasificar-archivo", files={"archivo": ("t.exe", b"x", "application/octet-stream")})
    assert r.status_code == 415
    r = cliente.post("/clasificar-archivo", files={"archivo": ("i.txt", TEXTO_INGLES.encode(), "text/plain")})
    assert r.status_code == 422 and r.json()["error"]["codigo"] == "idioma_no_espanol"


def test_pagina_de_inicio(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and "Detector de texto" in r.text


def test_documento_con_bloques_en_ingles_no_se_rechaza_completo(cliente):
    mezcla = TEXTO_INGLES * 3 + " " + TEXTO_VALIDO * 3  # el global tiene mucho texto en inglés
    r = cliente.post("/clasificar-archivo", files={"archivo": ("m.txt", mezcla.encode(), "text/plain")})
    assert r.status_code == 200 and r.json()["palabras_analizadas"] > 100
