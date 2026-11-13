#!/usr/bin/env bash
# Instala lo necesario (solo la primera vez), inicia el servicio y muestra la dirección.
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
echo "Servicio en http://127.0.0.1:8000  (Ctrl+C para detenerlo)"
uvicorn app.main:app
