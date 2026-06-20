#!/usr/bin/env python3
"""
Quantize a model to FP8 dynamic (llm-compressor → vLLM) — quant only, no prune.

FP8 = 8 bits/weight. For GLM-5.2 (753B) ≈ 753 GB → does NOT fit 560 GB.
See the loud warning the script prints. For models that DO fit (e.g.
DeepSeek-V2.5 236B ≈ 236 GB), this produces a clean vLLM-servable artifact.

NOTE: zai-org already ships `zai-org/GLM-5.2-FP8` — for GLM-5.2 you usually
just DOWNLOAD that rather than re-quantizing from BF16. This script exists for
other models and for reproducibility.

Usage:
    python quant_fp8.py --model-id deepseek-ai/DeepSeek-V2.5 --out ./out/dsv25-fp8
"""
import argparse
from pathlib import Path

import torch  # noqa: F401  (ensures clear error if env lacks torch)
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

from quant_common import resolve_prune_ignore, warn_753b_fit, print_vllm_serve


def main() -> None:
    p = argparse.ArgumentParser(description="FP8 dynamic quantization (no pruning)")
    p.add_argument("--model-id", default="zai-org/GLM-5.2")
    p.add_argument("--out", type=Path, default=Path("./out/fp8"))
    p.add_argument("--arch-preset", default="mla_moe",
                   choices=["mla_moe", "llama"],
                   help="Drives ignore patterns (MLA latents / MoE router stay high-precision)")
    p.add_argument("--offload-gb", type=int, default=0,
                   help="vLLM --cpu-offload-gb to print in the serve command")
    args = p.parse_args()

    warn_753b_fit(args.model_id, fmt_bits=8)
    args.out.mkdir(parents=True, exist_ok=True)

    # FP8_DYNAMIC needs no calibration data (activation scales computed at runtime).
    ignore = resolve_prune_ignore(args.arch_preset, ["lm_head"], moe=True, moe_ignore_router=True)
    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)

    print(f"[fp8] loading {args.model_id} (trust_remote_code — glm_moe_dsa native here)")
    oneshot(
        model=args.model_id,
        recipe=recipe,
        output_dir=str(args.out),
        trust_remote_code_model=True,
        oneshot_device="auto",
    )
    AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True).save_pretrained(args.out)
    print(f"\n✅ FP8 artifact → {args.out}")
    print_vllm_serve(args.out, "FP8 dynamic", offload_gb=args.offload_gb)


if __name__ == "__main__":
    main()
