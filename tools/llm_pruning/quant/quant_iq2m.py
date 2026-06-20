#!/usr/bin/env python3
"""
Quantize a model to GGUF IQ2_M (llama.cpp) — quantization only, no pruning.

IQ2_M ≈ 2.7 bits/weight. For GLM-5.2 (753B) ≈ 255 GB — fits the 2x 3090 +
512 GB DDR4 box (560 GB usable) with comfortable headroom. imatrix is REQUIRED
at 2-bit for usable quality (on by default here).

Usage:
    python quant_iq2m.py --model-id zai-org/GLM-5.2 \\
        --llama-cpp-dir /opt/llama.cpp --out ./out/glm52-iq2m

    # Fallback if llama.cpp lacks glm_moe_dsa support:
    python quant_iq2m.py --model-id deepseek-ai/DeepSeek-V3 \\
        --llama-cpp-dir /opt/llama.cpp --out ./out/dsv3-iq2m
"""
import argparse
from pathlib import Path

from quant_common import run_gguf_quant

FORMAT = "IQ2_M"


def main() -> None:
    p = argparse.ArgumentParser(description=f"GGUF {FORMAT} quantization (no pruning)")
    p.add_argument("--model-id", default="zai-org/GLM-5.2", help="HF id or local path")
    p.add_argument("--out", type=Path, default=Path("./out/iq2m"))
    p.add_argument("--llama-cpp-dir", type=Path, default=Path("/opt/llama.cpp"))
    p.add_argument("--moe", dest="moe", action="store_true", default=True,
                   help="Model is MoE (default true; affects serve command)")
    p.add_argument("--dense", dest="moe", action="store_false",
                   help="Model is dense (changes serve command to -ngl offload)")
    p.add_argument("--imatrix-samples", type=int, default=512)
    p.add_argument("--no-imatrix", dest="imatrix", action="store_false", default=True,
                   help="Skip imatrix (NOT recommended at 2-bit)")
    p.add_argument("--keep-f16", action="store_true",
                   help="Keep the f16 intermediate GGUF")
    args = p.parse_args()

    run_gguf_quant(
        model_id=args.model_id,
        out_dir=args.out,
        llama_cpp_dir=args.llama_cpp_dir,
        fmt=FORMAT,
        use_imatrix=args.imatrix,
        is_moe=args.moe,
        imatrix_samples=args.imatrix_samples,
        keep_f16=args.keep_f16,
    )


if __name__ == "__main__":
    main()
