#!/usr/bin/env python3
"""
Quantize a model to GGUF Q4_K_M (llama.cpp) — quantization only, no pruning.

Q4_K_M ≈ 4.8 bits/weight. For GLM-5.2 (753B) ≈ 452 GB — fits the 2x 3090 +
512 GB DDR4 box (560 GB usable). NOTE: at the old 256 GB this needed NVMe
paging; the 512 GB upgrade is what makes Q4_K_M run in RAM. Highest quality of
the three GGUF options. imatrix is OPTIONAL here (Q4 is robust without it) but
recommended — pass --imatrix to enable.

Usage:
    python quant_q4km.py --model-id zai-org/GLM-5.2 \\
        --llama-cpp-dir /opt/llama.cpp --out ./out/glm52-q4km --imatrix
"""
import argparse
from pathlib import Path

from quant_common import run_gguf_quant

FORMAT = "Q4_K_M"


def main() -> None:
    p = argparse.ArgumentParser(description=f"GGUF {FORMAT} quantization (no pruning)")
    p.add_argument("--model-id", default="zai-org/GLM-5.2", help="HF id or local path")
    p.add_argument("--out", type=Path, default=Path("./out/q4km"))
    p.add_argument("--llama-cpp-dir", type=Path, default=Path("/opt/llama.cpp"))
    p.add_argument("--moe", dest="moe", action="store_true", default=True)
    p.add_argument("--dense", dest="moe", action="store_false")
    p.add_argument("--imatrix-samples", type=int, default=512)
    # Q4_K_M is robust without imatrix → default OFF, opt-in via --imatrix.
    p.add_argument("--imatrix", dest="imatrix", action="store_true", default=False,
                   help="Enable imatrix-guided quant (recommended, slower)")
    p.add_argument("--keep-f16", action="store_true")
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
