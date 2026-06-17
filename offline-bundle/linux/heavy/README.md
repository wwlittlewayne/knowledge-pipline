# Heavy tier — OCR & video (optional, Linux)

The **core** Linux bundle (`../python/`) leaves out the OCR/video stack because it is huge and
awkward to vendor in git:

| Package | Purpose | Why it's not pre-vendored |
|---|---|---|
| `paddleocr` | image OCR (Chinese + multilingual) — `backend/processors/image_processor.py` | pulls `paddlepaddle` + a big native tree |
| `paddlepaddle` | OCR engine backend | single wheel often **> 100 MB** → exceeds GitHub's per-file limit (would need Git LFS) |
| `opencv-python` | video frame extraction — `backend/processors/video_processor.py` | ~60–90 MB; only needed for video |

All three are **lazy-imported**, so the pipeline runs fine without them — image OCR and video
processing simply log an "optional dependency missing" notice.

> Cross-building these from a non-matching box is unreliable (PaddlePaddle's Linux wheel tags vary),
> which is why this is a **download-on-a-matching-Linux-box** script rather than committed wheels.

## How to use (offline)

**Step 1 — on a Linux box matching the target** (CPython 3.9, x86_64, glibc) **WITH internet**:

```bash
cd offline-bundle/linux/heavy
bash download-ocr.sh            # OCR only
bash download-ocr.sh --opencv   # OCR + video
```

This downloads wheels into `heavy/wheelhouse/` and rewrites `requirements-heavy.txt` with exact pins.

**Step 2 — copy** the whole `offline-bundle/linux/heavy` folder (now containing `wheelhouse/`) to
the offline machine, into the same repo location.

**Step 3 — install offline.** Easiest: re-run the main installer — it auto-detects and installs the
heavy tier if `heavy/wheelhouse/` is present:

```bash
bash offline-bundle/linux/install-offline.sh
```

Or install manually into the project venv:

```bash
<repo>/.venv/bin/python -m pip install --no-index \
  --find-links offline-bundle/linux/heavy/wheelhouse \
  -r offline-bundle/linux/heavy/requirements-heavy.txt
```

## Troubleshooting

- **"Could not find a version that satisfies paddlepaddle"** — the newest PaddlePaddle may have
  dropped Python 3.9 (or your glibc). Pin an older one:
  ```bash
  bash download-ocr.sh --paddle 2.6.1     # try 2.6.x / 2.5.x for Python 3.9
  ```
- **First OCR run downloads models** — PaddleOCR fetches its detection/recognition models over the
  network on first use. On a truly air-gapped box, run it once on the internet machine to populate
  `~/.paddleocr/`, then copy that folder over too.
- `heavy/wheelhouse/` is **git-ignored** on purpose — don't commit these large wheels.
