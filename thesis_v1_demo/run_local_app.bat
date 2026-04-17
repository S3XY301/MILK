@echo off
cd /d "%~dp0"
set "VENV_PY=..\.venv-tfdml\Scripts\python.exe"
if exist "%VENV_PY%" (
  "%VENV_PY%" desktop_app.py
) else (
  python desktop_app.py
)
pause
