"""API: GET /salud y POST /clasificar. Documentación interactiva en /docs.
Arranque: uvicorn app.main:app"""
import os

import joblib
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import documentos, servicio


class Solicitud(BaseModel):
    texto: str


def crear_app(modelo=None, lm=None, modelo_path=None):
    app = FastAPI(title="Detector de texto generado por IA (prototipo)", version="0.1")
    estado = {"modelo": modelo, "lm": lm, "intentado": modelo is not None}

    def cargar():
        if not estado["intentado"]:
            estado["intentado"] = True
            ruta = modelo_path or os.getenv("MODEL_PATH", "models/modelo.joblib")
            if os.path.exists(ruta):
                estado["modelo"] = joblib.load(ruta)
        m = estado["modelo"]
        if m is not None and estado["lm"] is None and "perplejidad_media" in m["features"] and os.getenv("USE_LM", "1") != "0":
            from .perplexity import PerplejidadLM
            estado["lm"] = PerplejidadLM()
        return m

    @app.exception_handler(servicio.ErrorApi)
    async def _error(_: Request, e: servicio.ErrorApi):
        return JSONResponse(status_code=e.http, content={"error": {"codigo": e.codigo, "mensaje": e.mensaje}})

    @app.exception_handler(RequestValidationError)
    async def _validacion(_: Request, e: RequestValidationError):
        return JSONResponse(status_code=422, content={"error": {"codigo": "solicitud_invalida", "mensaje": "Falta el campo «texto»."}})

    @app.get("/salud")
    def salud():
        return {"estado": "activo", "modelo_cargado": cargar() is not None}

    @app.post("/clasificar")
    def clasificar(s: Solicitud):
        m = cargar()
        return servicio.clasificar(s.texto, m, estado["lm"])

    @app.post("/clasificar-archivo")
    def clasificar_archivo(archivo: UploadFile = File(...)):
        # def (no async): FastAPI lo ejecuta en un hilo y el servidor sigue respondiendo mientras analiza.
        print(f"[documento] recibido: {archivo.filename}", flush=True)
        texto = documentos.leer_documento(archivo.filename, archivo.file.read())
        print(f"[documento] leído: {len(texto.split())} palabras; cargando modelo...", flush=True)
        modelo = cargar()
        print("[documento] modelo listo; analizando...", flush=True)
        r = servicio.clasificar_documento(texto, modelo, estado["lm"])
        print(f"[documento] listo en {r['tiempo_ms'] / 1000:.1f} s", flush=True)
        return r

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def inicio():
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app


app = crear_app()
