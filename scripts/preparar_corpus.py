"""Prepara el corpus: recorta un fragmento de cada trabajo de grado (PDF) y limpia los textos de IA.

Uso (desde la raíz del repositorio):
    pip install pypdf
    python -m scripts.preparar_corpus            (toma la descripción del problema; --seccion introduccion para la introducción)

Carpetas de entrada (no se suben a GitHub):
    data/fuentes/humanos/   PDF de trabajos de grado, nombrados  H01_T01.pdf  (id H01, tema T01).
                            Si prefiere copiar el fragmento a mano, guárdelo como H01_T01.txt (UTF-8): solo se le quitan las citas.
    data/fuentes/ia/        textos de IA en .txt o .pdf, nombrados  I01_T01_chatgpt.txt  (id I01, tema T01, asistente chatgpt)

Salida:
    data/corpus/textos/<id>.txt   un texto por archivo (UTF-8)
    data/corpus/metadatos.csv     id,clase,autor,grupo,archivo   (lo lee scripts.extraer_caracteristicas)
    data/ficha_corpus.csv         ficha para completar a mano (enlace, año, versión del asistente, fecha)

El grupo es el tema: los textos de un mismo tema nunca quedan a la vez en entrenamiento y prueba.
Después de correr el script, ABRA cada texto, quite nombres propios o referencias sueltas y deje el archivo limpio.
"""
import argparse
import csv
import os
import re
import sys

OBJETIVO, MINIMO, MAXIMO = 350, 250, 500
SECCIONES = {
    "problema": r"(planteamiento|descripci[oó]n|definici[oó]n) del problema|problema de investigaci[oó]n|situaci[oó]n problem[aá]tica",
    "introduccion": r"introducci[oó]n",
    "justificacion": r"justificaci[oó]n",
}


def encabezado(nombre):
    return re.compile(r"^\s*(\d+(\.\d+)*\.?\s*)?(" + SECCIONES[nombre] + r")\s*$", re.I)


CITA_PARENTESIS = re.compile(r"\s*\([^()]*\b(?:19|20)\d{2}[a-z]?\b[^()]*\)")
CITA_CORCHETES = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")
URL_O_CORREO = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)")
NOMBRE_HUMANO = re.compile(r"^(H\d+)_(T\d+)\.(?:pdf|txt)$", re.I)
NOMBRE_IA = re.compile(r"^(I\d+)_(T\d+)_([A-Za-z0-9\-]+)\.(?:txt|pdf)$", re.I)


def contar(texto):
    return len(re.findall(r"\w+", texto, re.UNICODE))


def quitar_citas(texto):
    """Quita citas del tipo (Autor, 2019), [12] y direcciones web, para que el modelo no aprenda 'tiene citas'."""
    texto = CITA_PARENTESIS.sub("", texto)
    texto = CITA_CORCHETES.sub("", texto)
    texto = URL_O_CORREO.sub("", texto)
    texto = re.sub(r"\s+([,.;:])", r"\1", texto)
    return re.sub(r"[ \t]+", " ", texto).strip()


def es_prosa(linea):
    linea = linea.strip()
    if contar(linea) < 5 or re.search(r"\.{4,}", linea) or linea.isupper():
        return False
    if re.match(r"^(figura|tabla|gr[aá]fica|fuente|nota|cuadro)\b", linea, re.I):
        return False
    letras = sum(c.isalpha() for c in linea)
    return letras / max(1, len(linea.replace(" ", ""))) >= 0.75


def leer_pdf(ruta, max_paginas=60):
    from pypdf import PdfReader
    lector = PdfReader(ruta)
    paginas = []
    for p in lector.pages[:max_paginas]:
        paginas.append(p.extract_text() or "")
    texto = "\n".join(paginas)
    return re.sub(r"-\n(?=[a-záéíóúñ])", "", texto)  # une palabras cortadas al final de la línea


def bloques_de_prosa(texto, enc):
    """Devuelve [(texto_del_bloque, viene_tras_un_encabezado_de_seccion)]. Un bloque es una racha de líneas de prosa."""
    bloques, actual, tras_encabezado, marca = [], [], False, False
    for linea in texto.split("\n"):
        l = linea.strip()
        if es_prosa(l) or (actual and contar(l) >= 2 and re.search(r"[.!?]$", l)):
            if not actual:
                marca = tras_encabezado
            actual.append(l)
            continue
        if actual:
            bloques.append((" ".join(actual), marca))
            actual, tras_encabezado = [], False
        if enc.match(l) and not re.search(r"\.{3,}|\s\d+$", l):
            tras_encabezado = True
    if actual:
        bloques.append((" ".join(actual), marca))
    return bloques


def fragmento(texto, nombre):
    """Elige un fragmento de prosa de unas 350 palabras (entre 250 y 500) que termina en fin de oración."""
    bloques = bloques_de_prosa(texto, encabezado(nombre))
    fuertes = [i for i, (b, m) in enumerate(bloques) if m and contar(b) >= 100]
    if fuertes:
        inicio, seccion = fuertes[0], f"tras el encabezado de la sección «{nombre}»"
    else:
        largos = [i for i, (b, _) in enumerate(bloques) if contar(b) >= 150]
        if not largos:
            return None, ""
        inicio, seccion = largos[len(largos) // 3], f"NO se encontró la sección «{nombre}»: bloque largo de prosa, revise de dónde viene"
    corrido = " ".join(b for b, _ in bloques[inicio:inicio + 6])
    corrido = quitar_citas(corrido)
    oraciones = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ¿¡])", corrido)
    salida, n = [], 0
    for o in oraciones:
        k = contar(o)
        if n >= OBJETIVO or n + k > MAXIMO:
            break
        salida.append(o)
        n += k
    return " ".join(salida), seccion


def limpiar_ia(texto):
    """Solo quita marcas de formato (Markdown); no cambia las palabras. Las citas se quitan igual que en los humanos."""
    lineas = [re.sub(r"^\s*(#+\s*|[-*•]\s+)", "", l) for l in texto.splitlines()]
    t = "\n".join(lineas).replace("**", "").replace("__", "")
    t = re.sub(r"\n{2,}", "\n\n", t).strip()
    return quitar_citas(t)


def posibles_nombres(texto):
    sin_inicio = re.sub(r"(^|[.!?]\s+)[A-ZÁÉÍÓÚÑ]\w*", " ", texto)
    return sorted(set(re.findall(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b", sin_inicio)))[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--humanos", default="data/fuentes/humanos")
    ap.add_argument("--ia", default="data/fuentes/ia")
    ap.add_argument("--seccion", choices=sorted(SECCIONES), default="problema",
                    help="sección de la que se toma el fragmento (por defecto: descripción del problema)")
    ap.add_argument("--salida", default="data/corpus")
    ap.add_argument("--ficha", default="data/ficha_corpus.csv")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.salida, "textos"), exist_ok=True)
    meta, ficha, avisos = [], [], []

    for nombre in sorted(os.listdir(a.humanos)) if os.path.isdir(a.humanos) else []:
        m = NOMBRE_HUMANO.match(nombre)
        if not m:
            avisos.append(f"Ignorado (nombre inválido, use H01_T01.pdf o H01_T01.txt): {nombre}")
            continue
        id_, tema = m.group(1).upper(), m.group(2).upper()
        ruta = os.path.join(a.humanos, nombre)
        try:
            if nombre.lower().endswith(".txt"):  # fragmento copiado a mano: solo se le quitan las citas
                with open(ruta, encoding="utf-8") as f:
                    texto, seccion = quitar_citas(f.read()), "copiado a mano"
            else:
                texto, seccion = fragmento(leer_pdf(ruta), a.seccion)
        except Exception as e:  # PDF dañado o protegido
            avisos.append(f"{nombre}: no se pudo leer ({e})")
            continue
        if not texto:
            avisos.append(f"{nombre}: no se encontró prosa (¿PDF escaneado?). Copie el fragmento a mano.")
            continue
        n = contar(texto)
        if seccion.startswith("NO se encontró"):
            avisos.append(f"{id_}: no encontré la sección «{a.seccion}»; el fragmento sale de otra parte. Cópielo a mano si prefiere.")
        if n < MINIMO:
            avisos.append(f"{id_}: solo {n} palabras; busque otro fragmento o alargue el texto a mano")
        nombres = posibles_nombres(texto)
        if nombres:
            avisos.append(f"{id_}: revise posibles nombres propios: {', '.join(nombres)}")
        with open(os.path.join(a.salida, "textos", f"{id_}.txt"), "w", encoding="utf-8") as f:
            f.write(texto + "\n")
        meta.append({"id": id_, "clase": "humano", "autor": id_, "grupo": tema, "archivo": f"{id_}.txt"})
        ficha.append({"id": id_, "clase": "humano", "tema": tema, "palabras": n, "origen": nombre, "seccion": seccion,
                      "enlace": "", "anio": "", "asistente": "", "version": "", "fecha": ""})

    for nombre in sorted(os.listdir(a.ia)) if os.path.isdir(a.ia) else []:
        m = NOMBRE_IA.match(nombre)
        if not m:
            avisos.append(f"Ignorado (nombre inválido, use I01_T01_chatgpt.txt o I01_T01_chatgpt.pdf): {nombre}")
            continue
        id_, tema, asistente = m.group(1).upper(), m.group(2).upper(), m.group(3).lower()
        ruta = os.path.join(a.ia, nombre)
        if nombre.lower().endswith(".pdf"):
            try:
                crudo = re.sub(r"(?<!\n)\n(?!\n)", " ", leer_pdf(ruta, 10))
            except Exception as e:
                avisos.append(f"{nombre}: no se pudo leer ({e})")
                continue
            avisos.append(f"{id_}: viene de un PDF; revise que no haya texto de la interfaz del chat (fecha, botones, su propia instrucción).")
        else:
            with open(ruta, encoding="utf-8") as f:
                crudo = f.read()
        texto = limpiar_ia(crudo)
        n = contar(texto)
        if contar(texto.split("\n\n")[0]) < 15:
            avisos.append(f"{id_}: empieza con una línea corta (¿título o frase de cortesía del asistente?). Revise el inicio.")
        if re.search(r"^\s*([-*•]|\d+[.)])\s", crudo, re.M):
            avisos.append(f"{id_}: el asistente usó listas aunque se pidió prosa continua. Anótelo en la ficha o regenere con la misma instrucción.")
        if not MINIMO <= n <= MAXIMO:
            avisos.append(f"{id_}: tiene {n} palabras (se espera de {MINIMO} a {MAXIMO}). Si regenera, use la misma instrucción y anótelo.")
        with open(os.path.join(a.salida, "textos", f"{id_}.txt"), "w", encoding="utf-8") as f:
            f.write(texto + "\n")
        meta.append({"id": id_, "clase": "ia", "autor": asistente, "grupo": tema, "archivo": f"{id_}.txt"})
        ficha.append({"id": id_, "clase": "ia", "tema": tema, "palabras": n, "origen": nombre, "seccion": "",
                      "enlace": "", "anio": "", "asistente": asistente, "version": "", "fecha": ""})

    with open(os.path.join(a.salida, "metadatos.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "clase", "autor", "grupo", "archivo"])
        w.writeheader()
        w.writerows(meta)
    if os.path.exists(a.ficha):
        print(f"(La ficha {a.ficha} ya existe y no se sobrescribe, para no perder lo que usted anotó.)")
    else:
        os.makedirs(os.path.dirname(a.ficha) or ".", exist_ok=True)
        with open(a.ficha, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ficha[0].keys()) if ficha else ["id"])
            w.writeheader()
            w.writerows(ficha)

    h = sum(r["clase"] == "humano" for r in meta)
    i = sum(r["clase"] == "ia" for r in meta)
    print(f"\nCorpus: {h} humanos + {i} de IA = {len(meta)} textos -> {a.salida}")
    for tema in sorted({r["grupo"] for r in meta}):
        print(f"  {tema}: {sum(r['grupo'] == tema and r['clase'] == 'humano' for r in meta)} humanos, "
              f"{sum(r['grupo'] == tema and r['clase'] == 'ia' for r in meta)} IA")
    if h != i:
        avisos.append(f"Las clases no están balanceadas ({h} humanos, {i} de IA).")
    if avisos:
        print("\nPara revisar:")
        for x in avisos:
            print(" -", x)
    print("\nSiguiente paso: abra cada archivo de data/corpus/textos, quite nombres propios y restos de formato, y guárdelo.")
    return 0 if meta else 1


if __name__ == "__main__":
    sys.exit(main())
