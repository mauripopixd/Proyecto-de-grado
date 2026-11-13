@echo off
rem Instala lo necesario (solo la primera vez), inicia el servicio y abre el navegador.
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
    echo Creando el entorno virtual...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Instalando dependencias (la primera vez puede tardar varios minutos)...
pip install -q -r requirements.txt
start "" cmd /c "timeout /t 6 >nul & start http://127.0.0.1:8000"
echo.
echo Servicio en http://127.0.0.1:8000  (cierre esta ventana o presione Ctrl+C para detenerlo)
uvicorn app.main:app
pause
