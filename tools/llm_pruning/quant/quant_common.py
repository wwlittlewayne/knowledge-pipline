#!/usr/bin/env python3
"""
Shared helpers for the quantization-only scripts.

These scripts do NO pruning and NO recovery fine-tuning — they take a model
straight off the Hub (or a local path) and emit a single quantized artifact.

Two runtime families:
  - GGUF (llama.cpp): IQ2_M, IQ3_XXS, Q4_K_M  → CPU-offload friendly, runs on
    the 2x 3090 + EPYC box (and the 2x 2080 Ti box).
  - compressed-tensors (vLLM): FP8, INT8 (W8A8) → GPU-resident; for the full
    753B GLM-5.2 these need ~753 GB and do NOT fit 560 GB — see README.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# glm_arch lives one directory up; make it importable when run from quant/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from glm_arch import resolve_prune_ignore  # noqa: E402


# ----- calibration text (for imatrix + INT8 activation scales) ---------------

def dump_calib_text(
    out_txt: Path,
    n_samples: int = 512,
    min_chars: int = 200,
    dataset: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
) -> Path:
    """Write a plain-text calibration corpus (one sample per blank-line block).

    llama-imatrix consumes a raw text file; llm-compressor consumes an HF
    dataset. This produces the text file; INT8 builds its own HF dataset.
    """
    from datasets import load_dataset

    raw = load_dataset(dataset, subset, split="train")
    texts = [r["text"] for r in raw if r.get("text") and len(r["text"]) > min_chars]
    texts = texts[:n_samples]
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n\n".join(texts), encoding="utf-8")
    print(f"[calib] wrote {len(texts)} samples → {out_txt}")
    return out_txt


def build_calib_dataset(
    n_samples: int = 256,
    seqlen: int = 2048,
    dataset: str = "wikitext",
    subset: str = "wikitext-2-raw-v1",
    model_id: str | None = None,
):
    """Tokenized HF dataset for llm-compressor (INT8 activation calibration)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    raw = load_dataset(dataset, subset, split="train")

    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=seqlen)

    ds = (
        raw.filter(lambda r: r.get("text") and len(r["text"]) > 200)
           .select(range(min(n_samples * 2, len(raw))))
           .map(tok, remove_columns=raw.column_names)
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) >= seqlen // 2)
    return ds.select(range(min(n_samples, len(ds)))), tokenizer


# ----- llama.cpp toolchain ---------------------------------------------------

def resolve_llama_cpp(llama_cpp_dir: Path) -> tuple[Path, Path, Path]:
    """Return (convert_script, quantize_bin, imatrix_bin); raise if missing."""
    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize = llama_cpp_dir / "build" / "bin" / "llama-quantize"
    imatrix = llama_cpp_dir / "build" / "bin" / "llama-imatrix"
    if not convert.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert}.\n"
            f"  git clone https://github.com/ggerganov/llama.cpp {llama_cpp_dir}"
        )
    if not quantize.exists():
        raise FileNotFoundError(
            f"llama-quantize not found at {quantize}.\n"
            f"  cd {llama_cpp_dir} && cmake -B build -DGGML_CUDA=ON && "
            f"cmake --build build -j"
        )
    return convert, quantize, imatrix


def check_gguf_arch_support(model_id: str) -> None:
    """Warn loudly if the model architecture may be unsupported by llama.cpp.

    GLM-5.2 uses `glm_moe_dsa` (MLA + DeepSeek Sparse Attention), which as of
    last check is not upstream in llama.cpp. We don't hard-fail (the converter
    is the real authority), but we surface the risk + fallback before a long
    download/convert.
    """
    risky = ("glm-5.2", "glm5.2", "glm_moe_dsa")
    if any(tok in model_id.lower() for tok in risky):
        print("=" * 72)
        print("⚠️  llama.cpp ARCH SUPPORT WARNING")
        print(f"   {model_id} likely uses `glm_moe_dsa` (MLA + DSA sparse attn).")
        print("   This may NOT be supported by upstream convert_hf_to_gguf.py yet.")
        print("   • Check: https://github.com/ggerganov/llama.cpp/issues")
        print("   • Fallback model (fully supported, similar tier):")
        print("       deepseek-ai/DeepSeek-V3   (671B MoE)")
        print("   If convert fails below, that's why — switch --model-id.")
        print("=" * 72)


def convert_hf_to_gguf_f16(convert_script: Path, model_path: str, out_dir: Path) -> Path:
    """HF checkpoint → GGUF f16 intermediate. Returns the f16 gguf path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = model_path.rstrip("/").split("/")[-1]
    f16 = out_dir / f"{safe}-f16.gguf"
    print(f"[gguf] convert HF → f16: {model_path} → {f16}")
    subprocess.run(
        ["python", str(convert_script), model_path,
         "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    return f16


def gen_imatrix(imatrix_bin: Path, f16_gguf: Path, calib_txt: Path, out_dir: Path) -> Path:
    """Generate an importance matrix — essential for usable 2-3 bit quants."""
    if not imatrix_bin.exists():
        raise FileNotFoundError(
            f"llama-imatrix not found at {imatrix_bin}. Build llama.cpp with "
            f"the full tool set (cmake --build build -j builds it by default)."
        )
    imatrix = out_dir / "imatrix.dat"
    print(f"[imatrix] computing importance matrix → {imatrix}")
    subprocess.run(
        [str(imatrix_bin), "-m", str(f16_gguf), "-f", str(calib_txt),
         "-o", str(imatrix), "--chunks", "128"],
        check=True,
    )
    return imatrix


def llama_quantize(
    quantize_bin: Path,
    f16_gguf: Path,
    out_gguf: Path,
    fmt: str,
    imatrix: Path | None = None,
) -> Path:
    """Quantize f16 GGUF → target K/IQ format, optionally imatrix-guided."""
    cmd = [str(quantize_bin)]
    if imatrix is not None:
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(f16_gguf), str(out_gguf), fmt]
    print(f"[quant] {fmt}: {f16_gguf.name} → {out_gguf.name}"
          + (f" (imatrix-guided)" if imatrix else ""))
    subprocess.run(cmd, check=True)
    return out_gguf


# ----- serve-command printers ------------------------------------------------

def print_gguf_serve(gguf: Path, is_moe: bool) -> None:
    """Arch-aware llama-server recipes for the 3090+EPYC and 2080 Ti boxes."""
    print()
    print(f"[serve] ship: rsync -av --progress {gguf} user@<host>:/models/")
    print()
    if is_moe:
        print("[serve] 2× 3090 24GB + 2× EPYC 75F3:")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print( "        -ngl 999 --n-cpu-moe 32 \\        # raise to free VRAM for KV")
        print( "        -c 16384 --parallel 1 \\")
        print( "        --threads 64 --numa distribute --mlock \\")
        print( "        --host 0.0.0.0 --port 8080")
        print()
        print("[serve] 2× 2080 Ti 22GB (less VRAM, no FA2 → more experts on CPU):")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print( "        -ngl 999 --n-cpu-moe 56 \\")
        print( "        -c 8192 --parallel 1 --threads $(nproc) --mlock \\")
        print( "        --host 0.0.0.0 --port 8080")
    else:
        print("[serve] 2× 3090 24GB + 2× EPYC 75F3 (dense, fits VRAM):")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print( "        -ngl 999 --split-mode row \\")
        print( "        -c 32768 --parallel 1 \\")
        print( "        --threads 64 --numa distribute --mlock \\")
        print( "        --host 0.0.0.0 --port 8080")
        print()
        print("[serve] 2× 2080 Ti 22GB (44GB total; if OOM drop -ngl below total layers):")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print( "        -ngl 999 --split-mode row \\")
        print( "        -c 8192 --parallel 1 --threads $(nproc) --mlock \\")
        print( "        --host 0.0.0.0 --port 8080")


def print_vllm_serve(out_dir: Path, quant_name: str, offload_gb: int = 0) -> None:
    print()
    print(f"[serve] vLLM ({quant_name}):")
    line = f"      vllm serve {out_dir} --tensor-parallel-size 2 --dtype bfloat16"
    if offload_gb:
        line += f" \\\n        --cpu-offload-gb {offload_gb}"
    print(line + " \\")
    print( "        --max-model-len 16384 --enable-prefix-caching --max-num-seqs 16")


def run_gguf_quant(
    model_id: str,
    out_dir: Path,
    llama_cpp_dir: Path,
    fmt: str,
    use_imatrix: bool,
    is_moe: bool,
    imatrix_samples: int = 512,
    keep_f16: bool = False,
) -> Path:
    """Full GGUF flow shared by quant_iq2m / quant_iq3xxs / quant_q4km.

    convert HF→f16 → (imatrix) → llama-quantize → print serve cmd.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    check_gguf_arch_support(model_id)
    convert, quantize, imatrix_bin = resolve_llama_cpp(Path(llama_cpp_dir))

    f16 = convert_hf_to_gguf_f16(convert, model_id, out_dir)

    imatrix = None
    if use_imatrix:
        calib_txt = dump_calib_text(out_dir / "calib.txt", n_samples=imatrix_samples)
        imatrix = gen_imatrix(imatrix_bin, f16, calib_txt, out_dir)

    safe = model_id.rstrip("/").split("/")[-1]
    final = out_dir / f"{safe}-{fmt.lower()}.gguf"
    llama_quantize(quantize, f16, final, fmt, imatrix=imatrix)

    if not keep_f16 and f16.exists():
        f16.unlink()
        print(f"[gguf] removed intermediate {f16.name} (pass --keep-f16 to retain)")

    print(f"\n✅ {fmt} artifact → {final}")
    print_gguf_serve(final, is_moe=is_moe)
    return final


def warn_753b_fit(model_id: str, fmt_bits: int) -> None:
    """Loud header for FP8/INT8 — these don't fit 753B on 560 GB."""
    if any(t in model_id.lower() for t in ("glm-5.2", "glm5.2")):
        approx = 753  # GB at 8 bits
        print("=" * 72)
        print(f"⚠️  FIT WARNING — {fmt_bits}-bit on GLM-5.2 (753B)")
        print(f"   ≈ {approx} GB. Does NOT fit 560 GB (512 RAM + 48 VRAM).")
        print( "   Options:")
        print( "     • Use a GGUF script instead (IQ2_M / IQ3_XXS / Q4_K_M all fit).")
        print( "     • Target a smaller model: --model-id deepseek-ai/DeepSeek-V2.5")
        print( "     • Run on a multi-GPU node with ≥768 GB aggregate memory.")
        print("=" * 72)
