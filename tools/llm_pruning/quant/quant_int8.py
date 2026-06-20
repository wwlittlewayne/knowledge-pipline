#!/usr/bin/env python3
"""
Quantize a model to INT8 W8A8 (llm-compressor → vLLM) — quant only, no prune.

W8A8 = INT8 weights + INT8 activations = 8 bits/weight. For GLM-5.2 (753B)
≈ 753 GB → does NOT fit 560 GB (see warning). For models that fit (e.g.
DeepSeek-V2.5 236B), this gives near-lossless quality with good GPU throughput.

Unlike FP8_DYNAMIC, W8A8 needs CALIBRATION DATA to compute static activation
scales (SmoothQuant-style), so this script loads a small WikiText calib set.

Usage:
    python quant_int8.py --model-id deepseek-ai/DeepSeek-V2.5 --out ./out/dsv25-int8
"""
import argparse
from pathlib import Path

import torch  # noqa: F401
from transformers import AutoTokenizer
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

from quant_common import (
    resolve_prune_ignore, warn_753b_fit, print_vllm_serve, build_calib_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser(description="INT8 W8A8 quantization (no pruning)")
    p.add_argument("--model-id", default="zai-org/GLM-5.2")
    p.add_argument("--out", type=Path, default=Path("./out/int8"))
    p.add_argument("--arch-preset", default="mla_moe",
                   choices=["mla_moe", "llama"])
    p.add_argument("--calib-samples", type=int, default=256)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--offload-gb", type=int, default=0)
    args = p.parse_args()

    warn_753b_fit(args.model_id, fmt_bits=8)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[int8] building calibration set ({args.calib_samples} samples)")
    calib, _ = build_calib_dataset(
        n_samples=args.calib_samples, seqlen=args.calib_seqlen, model_id=args.model_id,
    )

    ignore = resolve_prune_ignore(args.arch_preset, ["lm_head"], moe=True, moe_ignore_router=True)
    # SmoothQuant migrates activation outliers into weights → better INT8 acts.
    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(targets="Linear", scheme="W8A8", ignore=ignore),
    ]

    print(f"[int8] loading {args.model_id} (trust_remote_code — glm_moe_dsa native here)")
    oneshot(
        model=args.model_id,
        dataset=calib,
        recipe=recipe,
        max_seq_length=args.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(args.out),
        trust_remote_code_model=True,
        oneshot_device="auto",
    )
    AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True).save_pretrained(args.out)
    print(f"\n✅ INT8 W8A8 artifact → {args.out}")
    print_vllm_serve(args.out, "INT8 W8A8", offload_gb=args.offload_gb)


if __name__ == "__main__":
    main()
