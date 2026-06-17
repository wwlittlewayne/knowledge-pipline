# Heavy tier — OCR & video (optional)

The **core** offline bundle (`../python/`) deliberately leaves out the OCR/video stack because
it is huge and awkward to vendor in git:

| Package | Purpose | Why it's not pre-vendored |
|---|---|---|
| `paddleocr` | image OCR (Chinese + multilingual) — `backend/processors/image_processor.py` | pulls `paddlepaddle` + a big native tree |
| `paddlepaddle` | OCR engine backend | single wheel often **> 100 MB** → exceeds GitHub's per-file limit (would need Git LFS) |
| `opencv-python` | video frame extraction — `backend/processors/video_processor.py` | ~60–90 MB; only needed for video |

All three are **lazy-imported**, so the pipeline runs fine without them — image OCR and video
processing simply log an "optional dependency missing" notice. Add this tier only if you need
OCR on scanned images or video ingestion.

> Note: cross-building these from Linux is unreliable (PaddlePaddle's wheels/tags differ per OS),
> which is exactly why this is a **download-on-Windows** script rather than pre-committed wheels.

## How to use (offline)

**Step 1 — on a Windows machine WITH internet** (CPython 3.9, 64-bit installed):

```powershell
cd offline-bundle\heavy
# OCR only:
powershell -ExecutionPolicy Bypass -File download-ocr.ps1
# OCR + video:
powershell -ExecutionPolicy Bypass -File download-ocr.ps1 -IncludeOpenCV
```

This downloads `win_amd64 / cp39` wheels into `heavy\wheelhouse\` and rewrites
`requirements-heavy.txt` with exact pins. (CMD users: `download-ocr.bat` / `download-ocr.bat opencv`.)

**Step 2 — copy** the whole `offline-bundle\heavy` folder (now containing `wheelhouse\`) to the
offline machine, into the same repo location.

**Step 3 — install offline.** Easiest: just re-run the main installer — it auto-detects and
installs the heavy tier if `heavy\wheelhouse\` is present:

```powershell
powershell -ExecutionPolicy Bypass -File offline-bundle\install-offline.ps1
```

Or install manually into the project venv:

```powershell
<repo>\.venv\Scripts\python -m pip install --no-index --find-links offline-bundle\heavy\wheelhouse -r offline-bundle\heavy\requirements-heavy.txt
```

## Troubleshooting

- **"Could not find a version that satisfies paddlepaddle (no win_amd64 wheel)"** — the newest
  PaddlePaddle may have dropped Python 3.9. Pin an older one that still ships a 3.9 Windows wheel:
  ```powershell
  powershell -ExecutionPolicy Bypass -File download-ocr.ps1 -PaddleVersion 2.6.1
  ```
  (Try `2.6.x` or `2.5.x`.)
- **First OCR run downloads models** — PaddleOCR fetches its recognition/detection models on first
  use over the network. On a truly air-gapped box, run it once on the internet machine to populate
  `%USERPROFILE%\.paddleocr\`, then copy that folder over too.
- `heavy\wheelhouse\` is **git-ignored** on purpose — don't commit these large wheels.
