@echo off
REM download-ocr.bat - CMD equivalent of download-ocr.ps1.
REM Run on a WINDOWS machine WITH internet + CPython 3.9 (64-bit) to fetch OCR wheels
REM into .\wheelhouse, then carry offline-bundle\heavy to the offline machine.
REM
REM Usage:   download-ocr.bat            (OCR only)
REM          download-ocr.bat opencv     (also download opencv-python for video)
setlocal EnableExtensions

set "HERE=%~dp0"
set "WHEELHOUSE=%HERE%wheelhouse"
if not exist "%WHEELHOUSE%" mkdir "%WHEELHOUSE%"

REM 1. pick launcher (prefer py -3.9)
set "PY="
where py >nul 2>nul && ( py -3.9 -c "import sys" >nul 2>nul && set "PY=py -3.9" )
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY ( echo [ERROR] Python not found. Install CPython 3.9.x 64-bit and re-run. & exit /b 1 )
%PY% -c "import sys;raise SystemExit(0 if sys.version_info[:2]==(3,9) else 1)" || echo [WARN] Python is not 3.9 - wheels will be tagged for the wrong version.

REM 2. package list
set "PKGS=paddlepaddle paddleocr"
if /I "%~1"=="opencv" set "PKGS=%PKGS% opencv-python"

echo Downloading heavy wheels: %PKGS%
echo (large - PaddlePaddle alone is ~100+ MB; needs internet)
%PY% -m pip download --only-binary=:all: --dest "%WHEELHOUSE%" %PKGS%
if errorlevel 1 (
  echo [WARN] A package lacked a win_amd64 wheel at the resolved version.
  echo        Try pinning, e.g.:  %PY% -m pip download --only-binary=:all: -d "%WHEELHOUSE%" paddlepaddle==2.6.1
  exit /b 1
)

echo.
echo Done. Wheels are in: %WHEELHOUSE%
echo Next on the OFFLINE machine: re-run offline-bundle\install-offline.bat (auto-installs heavy if present),
echo  or: ^<repo^>\.venv\Scripts\python -m pip install --no-index --find-links "%WHEELHOUSE%" paddleocr
endlocal
