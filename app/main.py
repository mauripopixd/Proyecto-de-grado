"""API: GET /salud y POST /clasificar. Documentación interactiva en /docs.
Arranque: uvicorn app.main:app"""
import os

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import servicio


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

    return app


app = crear_app()
