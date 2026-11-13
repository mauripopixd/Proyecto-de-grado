"""Envía uno o varios .txt a la API local y mide el tiempo de respuesta.
Uso (con la API encendida):  python probar_api.py data\\corpus\\textos\\H04.txt data\\corpus\\textos\\I04.txt"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8000/clasificar"

for ruta in sys.argv[1:]:
    texto = open(ruta, encoding="utf-8").read()
    pet = urllib.request.Request(URL, data=json.dumps({"texto": texto}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(pet) as r:
            cuerpo = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        cuerpo = e.read().decode("utf-8")
    seg = time.time() - t0
    print(f"\n{ruta}  ({len(texto.split())} palabras)  -> {seg:.1f} s")
    print(json.dumps(json.loads(cuerpo), ensure_ascii=False, indent=2))
