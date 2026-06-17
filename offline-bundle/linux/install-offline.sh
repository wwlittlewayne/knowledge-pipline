#!/usr/bin/env bash
#
# install-offline.sh — Offline installer for the knowledge-pipline FULL document pipeline (Linux).
#
# Installs all Python (and optional Node) dependencies from the vendored bundle with NO internet.
# Designed for a local LLM (Ollama / OpenAI-compatible).
#
# Target runtime: CPython 3.9 / x86_64 / glibc (manylinux). The vendored wheels are cp39 —
# they will NOT install on another Python minor version.
#
# Usage (from anywhere):
#   bash offline-bundle/linux/install-offline.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
WHEELHOUSE="$SCRIPT_DIR/python/wheelhouse"
REQS="$SCRIPT_DIR/python/requirements-offline.txt"
VENV="$REPO/.venv"

echo "knowledge-pipline — offline install (Linux)"
echo "repo root : $REPO"
echo

# 1. Resolve Python 3.9 (prefer the exact minor so the cp39 wheels match)
if command -v python3.9 >/dev/null 2>&1; then PY="python3.9"
elif command -v python3 >/dev/null 2>&1; then PY="python3"
else echo "ERROR: python3 not found. Install CPython 3.9 (x86_64) and re-run." >&2; exit 1; fi

ver="$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')"
if [ "$ver" != "3.9" ]; then
  echo "WARNING: selected Python is $ver, but the vendored wheels are cp39 (3.9)." >&2
  echo "         Install Python 3.9 (x86_64) or the native packages will fail to install." >&2
fi

# 2. Fresh venv at the repo root
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
VPY="$VENV/bin/python"

# 3. The offline install — no index, only the local wheelhouse
echo "Installing Python packages from $WHEELHOUSE ..."
"$VPY" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQS"

# 3b. OPTIONAL heavy tier (OCR/video) — only if wheels were generated via heavy/download-ocr.sh
HWH="$SCRIPT_DIR/heavy/wheelhouse"
if compgen -G "$HWH/*.whl" >/dev/null 2>&1; then
  echo "Found heavy OCR/video wheels — installing them too ..."
  "$VPY" -m pip install --no-index --find-links "$HWH" -r "$SCRIPT_DIR/heavy/requirements-heavy.txt" \
    || echo "WARNING: heavy tier install failed; core pipeline is still usable." >&2
fi

# 4. LLM config — copy the shared template to the live name if absent (never overwrite)
CFG="$REPO/.llm_config.json"
if [ ! -f "$CFG" ]; then
  cp "$SCRIPT_DIR/../llm_config.template.json" "$CFG"
  echo "Wrote .llm_config.json (Ollama default: http://localhost:11434/v1, model llama3.2)."
  echo "Edit it to match the model you 'ollama pull'-ed."
else
  echo ".llm_config.json already exists — left unchanged."
fi

# 5. OPTIONAL Node: restore the shared vendored node_modules (pptxgenjs)
NODE_TGZ="$SCRIPT_DIR/../node/node_modules.tgz"
if [ -f "$NODE_TGZ" ] && command -v tar >/dev/null 2>&1; then
  echo "Restoring vendored node_modules ..."
  tar -xzf "$NODE_TGZ" -C "$REPO"
fi

# 6. OPTIONAL register the /pipeline-* slash commands (needs node)
if command -v node >/dev/null 2>&1; then
  node "$REPO/scripts/install-commands.mjs" || echo "WARNING: slash-command registration skipped." >&2
fi

echo
echo "Done."
echo "Activate venv : source $VENV/bin/activate"
echo "Verify config : $VPY tools/test_llm_config.py   (start 'ollama serve' first)"
