<#
  install-offline.ps1 — Offline installer for the knowledge-pipline FULL document pipeline.

  Installs all Python (and optional Node) dependencies from the vendored bundle with
  NO internet access. Designed for a local LLM (Ollama / OpenAI-compatible).

  Target runtime: CPython 3.9.x (64-bit, win_amd64). The vendored wheels are cp39 —
  they will NOT install on another Python minor version.

  Usage (from anywhere):
    powershell -ExecutionPolicy Bypass -File offline-bundle\install-offline.ps1
#>

#requires -Version 5
$ErrorActionPreference = "Stop"

$BundleDir  = $PSScriptRoot
$RepoRoot   = Split-Path $BundleDir -Parent
$Wheelhouse = Join-Path $BundleDir "python\wheelhouse"
$Reqs       = Join-Path $BundleDir "python\requirements-offline.txt"
$VenvDir    = Join-Path $RepoRoot ".venv"

Write-Host "knowledge-pipline — offline install" -ForegroundColor Cyan
Write-Host "repo root : $RepoRoot"
Write-Host ""

# 1. Resolve Python 3.9 (prefer the py launcher, which can pin the minor version)
$PyExe = $null; $PyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    try { & py -3.9 -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $PyExe = "py"; $PyArgs = @("-3.9") } } catch {}
}
if (-not $PyExe -and (Get-Command python -ErrorAction SilentlyContinue)) { $PyExe = "python" }
if (-not $PyExe) { throw "Python not found. Install CPython 3.9.x (64-bit) from python.org and re-run." }

# 2. Verify it really is 3.9 (cp39 wheels require it)
$ver = & $PyExe @PyArgs -c "import sys;print('%d.%d'%sys.version_info[:2])"
if ($ver -ne "3.9") {
    Write-Warning "Selected Python is $ver, but the vendored wheels are built for 3.9 (win_amd64)."
    Write-Warning "Install Python 3.9.x (64-bit) and re-run, or native packages will fail to install."
}

# 3. Fresh venv
if (-not (Test-Path $VenvDir)) { & $PyExe @PyArgs -m venv $VenvDir }
$VenvPy = Join-Path $VenvDir "Scripts\python.exe"

# 4. The offline install (heart of it) — no index, only the local wheelhouse
Write-Host "Installing Python packages from $Wheelhouse ..." -ForegroundColor Cyan
& $VenvPy -m pip install --no-index --find-links $Wheelhouse -r $Reqs
if ($LASTEXITCODE -ne 0) { throw "Offline pip install failed (see messages above)." }

# 4b. OPTIONAL heavy tier (OCR/video) — install only if wheels were generated via heavy\download-ocr.ps1
$HeavyWh  = Join-Path $BundleDir "heavy\wheelhouse"
$HeavyReq = Join-Path $BundleDir "heavy\requirements-heavy.txt"
if ((Test-Path $HeavyWh) -and (Get-ChildItem $HeavyWh -Filter *.whl -ErrorAction SilentlyContinue)) {
    Write-Host "Found heavy OCR/video wheels — installing them too ..." -ForegroundColor Cyan
    & $VenvPy -m pip install --no-index --find-links $HeavyWh -r $HeavyReq
    if ($LASTEXITCODE -ne 0) { Write-Warning "Heavy tier install failed; core pipeline is still usable." }
}

# 5. LLM config — copy template to live name if absent (do NOT overwrite an existing one)
$Cfg = Join-Path $RepoRoot ".llm_config.json"
if (-not (Test-Path $Cfg)) {
    Copy-Item (Join-Path $BundleDir "llm_config.template.json") $Cfg
    Write-Host "Wrote .llm_config.json (Ollama default: http://localhost:11434/v1, model llama3.2)."
    Write-Host "Edit it to match the model you 'ollama pull'-ed."
} else {
    Write-Host ".llm_config.json already exists — left unchanged."
}

# 6. OPTIONAL Node: restore vendored node_modules (pptxgenjs) if npm/node available
$NodeTgz = Join-Path $BundleDir "node\node_modules.tgz"
if ((Test-Path $NodeTgz) -and (Get-Command tar -ErrorAction SilentlyContinue)) {
    Write-Host "Restoring vendored node_modules ..."
    & tar -xzf $NodeTgz -C $RepoRoot
}

# 7. OPTIONAL register the /pipeline-* slash commands (needs node)
if (Get-Command node -ErrorAction SilentlyContinue) {
    try { & node (Join-Path $RepoRoot "scripts\install-commands.mjs") } catch { Write-Warning "Slash-command registration skipped: $_" }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Activate venv : $VenvDir\Scripts\Activate.ps1"
Write-Host "Verify config : $VenvPy tools\test_llm_config.py   (start 'ollama serve' first)"
