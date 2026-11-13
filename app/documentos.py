"""Lectura de documentos (.txt, .pdf, .docx) y división en bloques. Todo ocurre en memoria: no se guarda nada."""
import io
import os
import re
import zipfile
from xml.etree import ElementTree as ET

from .servicio import ErrorApi

MAX_BYTES = 10 * 1024 * 1024
BLOQUE = 500          # palabras por bloque (el modelo se entrenó con fragmentos de 250 a 500 palabras)
MIN_BLOQUE = 100      # un bloque final más corto se une al anterior
MAX_BLOQUES = 10      # se analizan como máximo los primeros 10 bloques (unas 5.000 palabras)
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _txt(datos):
    for codificacion in ("utf-8-sig", "cp1252"):
        try:
            return datos.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return datos.decode("latin-1")


def _pdf(datos):
    try:
        from pypdf import PdfReader
        lector = PdfReader(io.BytesIO(datos))
        if lector.is_encrypted:
            raise ErrorApi("archivo_protegido", "El PDF está protegido con contraseña.")
        return "\n".join((p.extract_text() or "") for p in lector.pages)
    except ErrorApi:
        raise
    except Exception:
        raise ErrorApi("archivo_ilegible", "No se pudo leer el PDF.")


def _docx(datos):
    try:
        with zipfile.ZipFile(io.BytesIO(datos)) as z:
            raiz = ET.fromstring(z.read("word/document.xml"))
    except Exception:
        raise ErrorApi("archivo_ilegible", "No se pudo leer el documento de Word.")
    parrafos = ["".join(t.text or "" for t in p.iter(_W + "t")) for p in raiz.iter(_W + "p")]
    return "\n".join(parrafos)


def leer_documento(nombre, datos):
    """Devuelve el texto de un archivo .txt, .pdf o .docx (con espacios normalizados)."""
    if len(datos) > MAX_BYTES:
        raise ErrorApi("archivo_grande", "El archivo supera los 10 MB.", 413)
    ext = os.path.splitext((nombre or "").lower())[1]
    if ext == ".txt":
        texto = _txt(datos)
    elif ext == ".pdf":
        texto = _pdf(datos)
    elif ext == ".docx":
        texto = _docx(datos)
    else:
        raise ErrorApi("formato_no_soportado", "Use un archivo .txt, .pdf o .docx.", 415)
    texto = re.sub(r"-\s*\n\s*", "", texto)   # palabras cortadas con guion al final de línea
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        raise ErrorApi("archivo_vacio", "No se encontró texto en el archivo (¿es un PDF escaneado?).")
    return texto


def dividir_bloques(texto):
    """Divide el texto en bloques de unas BLOQUE palabras. Devuelve (bloques, truncado)."""
    pal = texto.split()
    bloques = [pal[i:i + BLOQUE] for i in range(0, len(pal), BLOQUE)]
    if len(bloques) > 1 and len(bloques[-1]) < MIN_BLOQUE:
        resto = bloques.pop()
        bloques[-1] += resto
    truncado = len(bloques) > MAX_BLOQUES
    return [" ".join(b) for b in bloques[:MAX_BLOQUES]], truncado
