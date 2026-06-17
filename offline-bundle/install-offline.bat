@echo off
REM install-offline.bat - Offline installer for knowledge-pipline full pipeline (CMD / double-click).
REM Target runtime: CPython 3.9.x (64-bit). Vendored wheels are cp39/win_amd64.
setlocal EnableExtensions

set "BUNDLE=%~dp0"
for %%I in ("%BUNDLE%..") do set "ROOT=%%~fI"
set "WHEELHOUSE=%BUNDLE%python\wheelhouse"
set "REQS=%BUNDLE%python\requirements-offline.txt"
set "VENV=%ROOT%\.venv"

echo knowledge-pipline - offline install
echo repo root : %ROOT%
echo.

REM 1. Pick a launcher (prefer py -3.9 to match the cp39 wheels)
set "PY="
where py >nul 2>nul && ( py -3.9 -c "import sys" >nul 2>nul && set "PY=py -3.9" )
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY ( echo [ERROR] Python not found. Install CPython 3.9.x 64-bit and re-run. & exit /b 1 )

REM 2. Warn if not 3.9
%PY% -c "import sys;raise SystemExit(0 if sys.version_info[:2]==(3,9) else 1)" || echo [WARN] Python is not 3.9 - the vendored native wheels (cp39) may fail to install.

REM 3. Fresh venv
if not exist "%VENV%" %PY% -m venv "%VENV%" || goto :err
set "VENVPY=%VENV%\Scripts\python.exe"

REM 4. Offline pip install
echo Installing Python packages from %WHEELHOUSE% ...
"%VENVPY%" -m pip install --no-index --find-links "%WHEELHOUSE%" -r "%REQS%" || goto :err

REM 4b. Optional heavy OCR/video tier (only if heavy\wheelhouse has wheels)
if exist "%BUNDLE%heavy\wheelhouse\*.whl" (
  echo Found heavy OCR/video wheels - installing them too ...
  "%VENVPY%" -m pip install --no-index --find-links "%BUNDLE%heavy\wheelhouse" -r "%BUNDLE%heavy\requirements-heavy.txt" || echo [WARN] Heavy tier install failed; core pipeline still usable.
)

REM 5. LLM config template -> live name (skip if present)
if not exist "%ROOT%\.llm_config.json" (
  copy /Y "%BUNDLE%llm_config.template.json" "%ROOT%\.llm_config.json" >nul
  echo Wrote .llm_config.json ^(Ollama default^). Edit model/base_url as needed.
) else (
  echo .llm_config.json already exists - left unchanged.
)

REM 6. Optional: restore vendored node_modules
if exist "%BUNDLE%node\node_modules.tgz" ( where tar >nul 2>nul && tar -xzf "%BUNDLE%node\node_modules.tgz" -C "%ROOT%" )

REM 7. Optional: register slash commands
where node >nul 2>nul && node "%ROOT%\scripts\install-commands.mjs"

echo.
echo Done.
echo Activate venv : %VENV%\Scripts\activate.bat
echo Verify config : "%VENVPY%" tools\test_llm_config.py   (start "ollama serve" first)
goto :eof

:err
echo [ERROR] Install failed. See messages above.
exit /b 1
