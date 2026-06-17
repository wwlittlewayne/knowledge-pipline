# Offline bundle — full pipeline with a local LLM (Windows)

> **On Linux?** Use [`linux/`](linux/README.md) instead (`bash offline-bundle/linux/install-offline.sh`).
> The files in *this* (root) folder are the **Windows** bundle. The `node/`, `vendor/`, and
> `llm_config.template.json` here are platform-neutral and shared by both.

Self-contained dependencies to run the **full document pipeline** (`/pipeline-ingest`,
`/pipeline-query`, `/pipeline-graph`, `/pipeline-ppt`) on a **Windows machine with no internet**,
driven by a **local LLM** (Ollama or any OpenAI-compatible server).

> Code Atlas (`/pipeline-code`) needs none of this — it's Python-stdlib only. This bundle is for
> the *document* pipeline, which needs third-party packages.

## What's inside

```
offline-bundle/
  install-offline.ps1 / .bat      # one-shot offline installer
  llm_config.template.json        # local-LLM config (copied to ../.llm_config.json)
  python/
    wheelhouse/                    # 53 vendored wheels (cp39 / win_amd64), ~32 MB
    requirements-offline.txt       # exact pins
    BUILD_MANIFEST.txt             # how/when the wheels were built
  node/node_modules.tgz            # vendored pptxgenjs (optional)
  vendor/                          # pptxgen.bundle.js + html2canvas.min.js (offline browser export)
  heavy/                           # OPTIONAL OCR/video — download-on-Windows scripts (see heavy/README.md)
```

## Prerequisites (install these on the target once)

- **CPython 3.9.x, 64-bit** — the vendored wheels are built for `cp39 / win_amd64`. They will
  **not** install on another Python minor version. Get it from python.org (tick "Add to PATH").
- **Node.js** (optional) — only needed to register the `/pipeline-*` slash commands and to restore
  the vendored `node_modules`. The Python pipeline + native PPTX export work without it.
- A **local LLM**, e.g. [Ollama](https://ollama.com): `ollama serve` then `ollama pull llama3.2`.

## Install (no internet required)

From the repo root, in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File offline-bundle\install-offline.ps1
```

(or double-click / run `offline-bundle\install-offline.bat`). It will:
1. find Python 3.9 (prefers `py -3.9`) and create a fresh `.venv` at the repo root,
2. `pip install --no-index --find-links python\wheelhouse -r python\requirements-offline.txt`,
3. (if present) install the optional heavy OCR/video wheels,
4. write `.llm_config.json` from the template (only if missing),
5. (optional) restore `node_modules` and register the slash commands if Node is present.

## Point it at your local LLM

The installer writes `.llm_config.json` (Ollama default):

```json
{ "base_url": "http://localhost:11434/v1", "model": "llama3.2", "api_key": "" }
```

Set `model` to whatever you `ollama pull`-ed (e.g. `qwen2`, `mixtral`); `api_key` stays empty for
Ollama. Any OpenAI-compatible local server works — just change `base_url`.

## Verify

```powershell
py -3.9 -V                                              # must say 3.9.x (64-bit)
.\.venv\Scripts\python -c "import openai,yaml,flask,flask_cors,pydantic,pptx,lxml,PIL,networkx,docx,openpyxl,pypdf,bs4,trafilatura; print('CORE OK')"
ollama serve                                            # in another terminal, then: ollama pull llama3.2
.\.venv\Scripts\python tools\test_llm_config.py         # checks config + LLM connectivity
.\.venv\Scripts\python tools\pipeline_ingest.py "C:\path\to\sample.pdf"
.\.venv\Scripts\python tools\pipeline_query.py "summarize the document"
```

## What works / what's optional

- **Works out of the box (core):** PDF, DOCX, XLSX, PPTX, HTML (via trafilatura/bs4), plain text;
  images handled via the local LLM's vision (if your model supports it); knowledge graph, query,
  lint, LivePPT server, and native `python-pptx` export.
- **Optional OCR / video** (scanned-image OCR, video frames): not included by default — see
  [`heavy/README.md`](heavy/README.md). Run `heavy\download-ocr.ps1` on a Windows box with internet
  to fetch PaddleOCR/OpenCV, then re-run the installer.
- **Offline browser PPTX export:** `ppt_live/index.html` normally loads pptxgenjs + html2canvas +
  Google Fonts from a CDN (dead offline). The native `python-pptx` export is unaffected. To restore
  the *browser* export offline, point those `<script>` tags at the vendored copies in `vendor/`
  (`pptxgen.bundle.js`, `html2canvas.min.js`); fonts fall back to system fonts.

## Rebuilding the wheelhouse

The wheels were cross-downloaded for `cp39 / win_amd64`. To rebuild (needs internet):

```bash
python3 -m pip download --only-binary=:all: --platform win_amd64 \
  --python-version 3.9 --implementation cp --abi cp39 \
  --dest offline-bundle/python/wheelhouse \
  openai requests networkx pypdf python-docx openpyxl python-pptx \
  beautifulsoup4 lxml pillow pyyaml python-dotenv flask flask-cors pydantic trafilatura
```

See `python/BUILD_MANIFEST.txt` for the exact provenance of the current set.
