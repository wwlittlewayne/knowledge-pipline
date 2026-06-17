# Offline bundle — full pipeline with a local LLM (Linux)

Self-contained dependencies to run the **full document pipeline** (`/pipeline-ingest`,
`/pipeline-query`, `/pipeline-graph`, `/pipeline-ppt`) on a **Linux machine with no internet**,
driven by a **local LLM** (Ollama or any OpenAI-compatible server).

> Code Atlas (`/pipeline-code`) needs none of this — it's Python-stdlib only. This bundle is for
> the *document* pipeline, which needs third-party packages.
>
> Windows? Use the bundle in the parent folder (`../install-offline.ps1`).

## What's inside

```
offline-bundle/
  llm_config.template.json        # shared local-LLM config  (reused by this installer)
  node/node_modules.tgz           # shared vendored pptxgenjs (reused)
  vendor/*.js                     # shared offline browser-export assets (reused)
  linux/
    install-offline.sh            # one-shot offline installer (this folder)
    python/
      wheelhouse/                 # 53 wheels (cp39 / manylinux x86_64), ~34 MB
      requirements-offline.txt    # exact pins
      BUILD_MANIFEST.txt          # how/when the wheels were built
    heavy/                        # OPTIONAL OCR/video — download-on-Linux scripts (see heavy/README.md)
```

## Prerequisites (install these on the target once)

- **CPython 3.9, x86_64, glibc** — the vendored wheels are built for `cp39 / manylinux_x86_64`
  (minimum **glibc 2.28**). They will **not** install on another Python minor version, another
  arch (aarch64), or musl/Alpine. Check with `python3.9 --version`, `uname -m`, `ldd --version`.
- **Node.js** (optional) — only to register the `/pipeline-*` slash commands and restore the
  vendored `node_modules`. The Python pipeline + native PPTX export work without it.
- A **local LLM**, e.g. [Ollama](https://ollama.com): `ollama serve` then `ollama pull llama3.2`.

## Install (no internet required)

From the repo root:

```bash
bash offline-bundle/linux/install-offline.sh
```

It will:
1. find Python 3.9 (prefers `python3.9`) and create a fresh `.venv` at the repo root,
2. `pip install --no-index --find-links python/wheelhouse -r python/requirements-offline.txt`,
3. (if present) install the optional heavy OCR/video wheels,
4. write `.llm_config.json` from the shared template (only if missing),
5. (optional) restore `node_modules` and register the slash commands if Node is present.

## Point it at your local LLM

The installer writes `.llm_config.json` (Ollama default):

```json
{ "base_url": "http://localhost:11434/v1", "model": "llama3.2", "api_key": "" }
```

Set `model` to whatever you `ollama pull`-ed; `api_key` stays empty for Ollama. Any
OpenAI-compatible local server works — just change `base_url`.

## Verify

```bash
python3.9 --version            # 3.9.x ;  uname -m -> x86_64 ;  ldd --version -> glibc >= 2.28
source .venv/bin/activate
python -c "import openai,yaml,flask,flask_cors,pydantic,pptx,lxml,PIL,networkx,docx,openpyxl,pypdf,bs4,trafilatura; print('CORE OK')"
ollama serve &                 # then: ollama pull llama3.2
python tools/test_llm_config.py
python tools/pipeline_ingest.py /path/to/sample.pdf
python tools/pipeline_query.py "summarize the document"
```

## What works / what's optional

- **Works out of the box (core):** PDF, DOCX, XLSX, PPTX, HTML (trafilatura/bs4), text; images via
  the local LLM's vision (if your model supports it); knowledge graph, query, lint, LivePPT server,
  native `python-pptx` export.
- **Optional OCR / video:** not included by default — see [`heavy/README.md`](heavy/README.md). Run
  `heavy/download-ocr.sh` on a matching Linux box with internet, then re-run the installer.
- **Offline browser PPTX export:** `ppt_live/index.html` loads pptxgenjs + html2canvas + Google
  Fonts from a CDN (dead offline). Native `python-pptx` export is unaffected. To restore the browser
  export, point those `<script>` tags at the vendored copies in `../vendor/`.

## Rebuilding the wheelhouse

Cross-built for `cp39 / manylinux x86_64` from a glibc Linux box (needs internet):

```bash
python3 -m pip download --only-binary=:all: --implementation cp --python-version 3.9 --abi cp39 \
  --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 \
  --dest offline-bundle/linux/python/wheelhouse \
  openai requests networkx pypdf python-docx openpyxl python-pptx beautifulsoup4 \
  lxml pillow pyyaml python-dotenv flask flask-cors pydantic trafilatura
```

See `python/BUILD_MANIFEST.txt` for the exact provenance of the current set.
