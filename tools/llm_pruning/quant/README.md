# Quantization-only scripts (no pruning, no recovery)

Take a model straight off the Hub and emit one quantized artifact. No
SparseGPT/Wanda pruning, no LoRA recovery — pure quantization. Built for the
2× 3090 + 2× EPYC 75F3 box after the **512 GB DDR4** upgrade (512 GB RAM +
48 GB VRAM = **560 GB usable**), so GLM-5.2 can run by direct quantization
instead of the multi-day prune+recover pipeline.

## Fit table — GLM-5.2 (753B params)

| Script | Format | Runtime | bits/wt | Approx size | Fits 560 GB? |
|---|---|---|---|---|---|
| `quant_iq2m.py` | IQ2_M | llama.cpp GGUF | ~2.7 | ~255 GB | ✅ comfortable |
| `quant_iq3xxs.py` | IQ3_XXS | llama.cpp GGUF | ~3.06 | ~309 GB | ✅ comfortable — **best quality/size**|
| `quant_q4km.py` | Q4_K_M | llama.cpp GGUF | ~4.8 | ~452 GB | ✅ fits (the 512 GB upgrade unlocks this) |
| `quant_fp8.py` | FP8 dynamic | vLLM | 8.0 | ~753 GB | ❌ needs ~768 GB+ |
| `quant_int8.py` | INT8 W8A8 | vLLM | 8.0+act | ~753 GB | ❌ needs ~768 GB+ |

**The three GGUF scripts are the runnable options for the full 753B on this
box.** FP8/INT8 of 753B do not fit even at 512 GB — those scripts print a loud
warning and work best on a smaller `--model-id` (e.g. DeepSeek-V2.5 236B ≈
236 GB) or a multi-GPU node.

> Earlier planning notes mis-stated FP8 as ~376 GB. Corrected: 753B × 1 byte
> = ~753 GB. The 256 routed experts × 75 MoE layers dominate the param count.

## ⚠️ llama.cpp architecture support (GGUF scripts)

GLM-5.2 uses `glm_moe_dsa` (MLA + DeepSeek Sparse Attention). As of last check
this is **not yet supported** by upstream `convert_hf_to_gguf.py`. The GGUF
scripts print a warning before the long convert step. If conversion fails:

- Check https://github.com/ggerganov/llama.cpp/issues for `glm_moe_dsa`
- Fallback model (fully supported, similar tier): `deepseek-ai/DeepSeek-V3`
  (671B MoE — IQ2_M ~228 GB / IQ3_XXS ~276 GB / Q4_K_M ~404 GB, all fit)

FP8/INT8 scripts go through `transformers` with `trust_remote_code=True`, so
`glm_moe_dsa` works there — the blocker for those two is memory, not support.

## Prerequisites

**Python 3.11 or 3.12** (3.12 recommended; floor 3.10, avoid 3.13). Use a
fresh venv/conda env per box.

Fastest path — the auto-detecting setup script (one dir up) installs the
arch-matched PyTorch + deps AND builds llama.cpp in one shot:

```bash
cd ..                 # tools/llm_pruning/
./setup.sh --llama-cpp-dir /opt/llama.cpp
```

Then run any quant script with `--llama-cpp-dir /opt/llama.cpp`.

> The scripts do NOT auto-install packages or build llama.cpp themselves —
> `setup.sh` does that. Model weights + the WikiText calibration set DO
> download automatically from HuggingFace on first run.

Manual equivalent:

```bash
# GGUF scripts — build llama.cpp
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j $(nproc)
pip install -r requirements.txt    # for convert_hf_to_gguf.py

# FP8/INT8 scripts
pip install -r ../requirements.txt  # llm-compressor, transformers, torch, datasets
```

## Usage

```bash
# IQ2_M — smallest, 2-bit, imatrix on by default
python quant_iq2m.py   --model-id zai-org/GLM-5.2 --llama-cpp-dir /opt/llama.cpp --out ./out/glm52-iq2m

# IQ3_XXS — recommended quality/size sweet spot
python quant_iq3xxs.py --model-id zai-org/GLM-5.2 --llama-cpp-dir /opt/llama.cpp --out ./out/glm52-iq3xxs

# Q4_K_M — highest GGUF quality; add --imatrix for best results
python quant_q4km.py   --model-id zai-org/GLM-5.2 --llama-cpp-dir /opt/llama.cpp --out ./out/glm52-q4km --imatrix

# FP8 — for a model that fits (or just download zai-org/GLM-5.2-FP8)
python quant_fp8.py    --model-id deepseek-ai/DeepSeek-V2.5 --out ./out/dsv25-fp8

# INT8 W8A8 — needs calibration; SmoothQuant applied automatically
python quant_int8.py   --model-id deepseek-ai/DeepSeek-V2.5 --out ./out/dsv25-int8
```

For **dense** models pass `--dense` to the GGUF scripts so the printed
`llama-server` command uses partial `-ngl <N>` offload instead of the MoE
`--n-cpu-moe` knob.

## imatrix (importance matrix)

2-bit and 3-bit quants need an imatrix for usable quality — it tells
`llama-quantize` which weights to protect. Defaults:

| Script | imatrix default | Flag |
|---|---|---|
| `quant_iq2m.py` | **on** | `--no-imatrix` to disable |
| `quant_iq3xxs.py` | **on** | `--no-imatrix` to disable |
| `quant_q4km.py` | off | `--imatrix` to enable (recommended) |

The imatrix is computed from WikiText calibration (`--imatrix-samples`, default
512). For domain work you can point `dump_calib_text` in `quant_common.py` at a
code/EE corpus, but WikiText is a safe general default.

## Quality vs size (MoE, approximate, vs BF16)

| Format | MMLU drop | HumanEval | GSM8K | Notes |
|---|---|---|---|---|
| Q4_K_M | −1 to −2 | −2% | −3% | safest; near-lossless for MoE |
| IQ3_XXS | −3 to −5 | −5% | −7% | best sub-4-bit tradeoff |
| IQ2_M | −5 to −8 | −8% | −12% | usable; noticeable on hard reasoning |

MoE tolerates low-bit better than dense (smaller active path, routing masks
local errors). For engineering+coding, **IQ3_XXS** is the recommended balance.

## Shared code

`quant_common.py` holds the calibration dump, imatrix generation, HF→GGUF
conversion, llama.cpp path resolution, the `run_gguf_quant` flow, and the
arch-aware serve-command printers. It reuses `../glm_arch.py`'s
`resolve_prune_ignore` so FP8/INT8 keep MLA latent projections and MoE router
weights in higher precision.
