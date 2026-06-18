#!/usr/bin/env python3
"""
Big-tier GLM-5 prune + recover + quantize pipeline.

Target hardware: RTX Pro 6000 Blackwell, 96GB (sm_100).
Native FP8/FP4, 2:4 sparse tensor-core acceleration, FlashAttention-3.

Produces a 70B-class (dense or MoE) GLM-5 with 50% 2:4 sparse weights + FP8
dynamic quantization, served via vLLM with FP8 KV cache and prefix caching
for multi-user throughput.

Stages (each saves to disk; resume from any stage via --stage / --*-path):
    1. sparsegpt : SparseGPT 2:4 semi-structured pruning  (per-layer CPU offload)
    2. recover   : QLoRA recovery fine-tune on the 70B sparse base
    3. merge     : Merge LoRA, re-apply 2:4 mask (preserves sparse pattern
                   broken by dense LoRA delta)
    4. fp8       : FP8 dynamic quantization, composes with 2:4 mask;
                   final artifact serves on Blackwell vLLM at ~2.5-3x BF16 tps.

Usage:
    python big_tier.py --config configs/big_tier.yaml --stage all
    python big_tier.py --stage fp8 --merged-path ./out/big_tier/merged_bf16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from glm_arch import resolve_lora_targets, resolve_prune_ignore


@dataclass
class Config:
    model_id: str = "zai-org/GLM-5.2-FP8"
    output_root: Path = Path("./out/big_tier")
    offload_folder: Path = Path("/tmp/llmcompressor_offload")

    # "mla_moe" for GLM-5.2 / DeepSeek V3; "llama" for dense models
    arch_preset: str = "mla_moe"

    sparsity: float = 0.5
    mask_structure: str = "2:4"
    calib_dataset: str = "wikitext"
    calib_subset: str = "wikitext-2-raw-v1"
    calib_samples: int = 512
    calib_seqlen: int = 2048

    recover_dataset: str = "Open-Orca/OpenOrca"
    recover_samples: int = 50_000
    lora_rank: int = 64
    lora_alpha: int = 128
    # Empty → use arch_preset's lora_targets. Non-empty list overrides.
    lora_target_modules: List[str] = field(default_factory=list)
    lora_epochs: int = 1
    lora_lr: float = 1e-4
    lora_batch: int = 1
    lora_grad_accum: int = 16
    lora_seqlen: int = 2048

    moe: bool = True
    moe_ignore_router: bool = True

    fp8_scheme: str = "FP8_DYNAMIC"
    # Empty → use arch_preset's prune_ignore. Non-empty list appends to it.
    fp8_ignore: List[str] = field(default_factory=list)

    remask_calib_samples: int = 128

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key in ("output_root", "offload_folder"):
            if key in data:
                data[key] = Path(data[key])
        return cls(**data)


def _build_calib(cfg: Config, n: int):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    raw = load_dataset(cfg.calib_dataset, cfg.calib_subset, split="train")

    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=cfg.calib_seqlen)

    ds = (
        raw.filter(lambda r: r.get("text") and len(r["text"]) > 200)
           .select(range(min(n * 2, len(raw))))
           .map(tok, remove_columns=raw.column_names)
    )
    ds = ds.filter(lambda x: len(x["input_ids"]) >= cfg.calib_seqlen // 2)
    return ds.select(range(min(n, len(ds)))), tokenizer


def _ignore_targets(cfg: Config) -> List[str]:
    return resolve_prune_ignore(
        cfg.arch_preset, cfg.fp8_ignore, cfg.moe, cfg.moe_ignore_router,
    )


def stage_sparsegpt(cfg: Config) -> Path:
    out = cfg.output_root / "sparsegpt"
    out.mkdir(parents=True, exist_ok=True)
    cfg.offload_folder.mkdir(parents=True, exist_ok=True)
    print(f"[sparsegpt] model={cfg.model_id} mask={cfg.mask_structure} "
          f"sparsity={cfg.sparsity:.0%} (offload={cfg.offload_folder})")

    calib, tokenizer = _build_calib(cfg, cfg.calib_samples)

    recipe = SparseGPTModifier(
        sparsity=cfg.sparsity,
        mask_structure=cfg.mask_structure,
        targets=["Linear"],
        ignore=_ignore_targets(cfg),
    )

    oneshot(
        model=cfg.model_id,
        dataset=calib,
        recipe=recipe,
        max_seq_length=cfg.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(out),
        trust_remote_code_model=True,
        oneshot_device="auto",
        offload_folder=str(cfg.offload_folder),
    )
    tokenizer.save_pretrained(out)
    print(f"[sparsegpt] saved → {out}")
    return out


def stage_recover(cfg: Config, pruned_path: Path) -> Path:
    out = cfg.output_root / "lora"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[recover] base={pruned_path} (QLoRA NF4 — 70B fits ~35GB in 96GB)")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        pruned_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(pruned_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=resolve_lora_targets(cfg.arch_preset, cfg.lora_target_modules),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    ds = (
        load_dataset(cfg.recover_dataset, split="train")
        .shuffle(seed=42)
        .select(range(cfg.recover_samples))
    )

    def to_text(ex):
        if "messages" in ex:
            return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
        user = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
        sys = ex.get("system_prompt") or ex.get("system") or ""
        bot = ex.get("output") or ex.get("response") or ex.get("answer") or ""
        msgs = []
        if sys:
            msgs.append({"role": "system", "content": sys})
        msgs += [
            {"role": "user", "content": user},
            {"role": "assistant", "content": bot},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    ds = ds.map(to_text, remove_columns=ds.column_names)

    sft_args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=cfg.lora_epochs,
        per_device_train_batch_size=cfg.lora_batch,
        gradient_accumulation_steps=cfg.lora_grad_accum,
        learning_rate=cfg.lora_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=cfg.lora_seqlen,
        packing=True,
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=base,
        args=sft_args,
        train_dataset=ds,
        peft_config=lora_cfg,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out))
    print(f"[recover] LoRA adapter saved → {out}")
    return out


def stage_merge(cfg: Config, pruned_path: Path, lora_path: Path) -> Path:
    """Merge LoRA into BF16 base, then re-apply 2:4 mask.

    The dense LoRA delta breaks the strict 2:4 pattern; we run a short
    SparseGPT pass to re-establish it. This is much faster than the initial
    prune (smaller calib set, model already near-sparse).
    """
    out = cfg.output_root / "merged_bf16"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] base={pruned_path} adapter={lora_path}")

    base = AutoModelForCausalLM.from_pretrained(
        pruned_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        offload_folder=str(cfg.offload_folder),
    )
    merged = PeftModel.from_pretrained(base, lora_path).merge_and_unload()

    pre_remask = out / "pre_remask"
    pre_remask.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(pre_remask), safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(pruned_path, trust_remote_code=True).save_pretrained(pre_remask)
    del base, merged
    torch.cuda.empty_cache()

    print(f"[merge] re-applying {cfg.mask_structure} mask via SparseGPT "
          f"({cfg.remask_calib_samples} calib samples)")
    calib, _ = _build_calib(cfg, cfg.remask_calib_samples)

    recipe = SparseGPTModifier(
        sparsity=cfg.sparsity,
        mask_structure=cfg.mask_structure,
        targets=["Linear"],
        ignore=_ignore_targets(cfg),
    )
    oneshot(
        model=str(pre_remask),
        dataset=calib,
        recipe=recipe,
        max_seq_length=cfg.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(out),
        trust_remote_code_model=True,
        oneshot_device="auto",
        offload_folder=str(cfg.offload_folder),
    )
    AutoTokenizer.from_pretrained(pruned_path, trust_remote_code=True).save_pretrained(out)
    print(f"[merge] merged + re-masked → {out}")
    return out


def stage_fp8(cfg: Config, merged_path: Path) -> Path:
    """FP8 dynamic quantization preserving the 2:4 sparse mask."""
    out = cfg.output_root / "fp8_24"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[fp8] model={merged_path} scheme={cfg.fp8_scheme}")

    calib, _ = _build_calib(cfg, cfg.calib_samples // 2)

    recipe = QuantizationModifier(
        targets="Linear",
        scheme=cfg.fp8_scheme,
        ignore=_ignore_targets(cfg),
    )

    oneshot(
        model=str(merged_path),
        dataset=calib,
        recipe=recipe,
        max_seq_length=cfg.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(out),
        trust_remote_code_model=True,
        oneshot_device="auto",
        offload_folder=str(cfg.offload_folder),
    )
    AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True).save_pretrained(out)
    print(f"[fp8] saved → {out}")
    print()
    print(f"[fp8] serve on Blackwell with vLLM:")
    print(f"      vllm serve {out} \\")
    print(f"        --dtype auto \\")
    print(f"        --kv-cache-dtype fp8 \\")
    print(f"        --max-model-len 32768 \\")
    print(f"        --enable-prefix-caching \\")
    print(f"        --max-num-seqs 256")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--stage", choices=["sparsegpt", "recover", "merge", "fp8", "all"],
                   default="all")
    p.add_argument("--pruned-path", type=Path)
    p.add_argument("--lora-path", type=Path)
    p.add_argument("--merged-path", type=Path)

    # Domain / dataset overrides — apply on top of YAML defaults.
    p.add_argument("--model-id",
                   help="HF model id, overrides Config.model_id")
    p.add_argument("--recover-dataset",
                   help="HF dataset for LoRA recovery (e.g. ise-uiuc/Magicoder-Evol-Instruct-110K for coding)")
    p.add_argument("--recover-samples", type=int)
    p.add_argument("--lora-epochs", type=int)
    p.add_argument("--output-root", type=Path,
                   help="Where stage artifacts land; isolate per-domain runs")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()

    if args.model_id:
        cfg.model_id = args.model_id
    if args.recover_dataset:
        cfg.recover_dataset = args.recover_dataset
    if args.recover_samples:
        cfg.recover_samples = args.recover_samples
    if args.lora_epochs:
        cfg.lora_epochs = args.lora_epochs
    if args.output_root:
        cfg.output_root = args.output_root

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    pruned = args.pruned_path or (cfg.output_root / "sparsegpt")
    lora = args.lora_path or (cfg.output_root / "lora")
    merged = args.merged_path or (cfg.output_root / "merged_bf16")

    if args.stage in ("sparsegpt", "all"):
        pruned = stage_sparsegpt(cfg)
    if args.stage in ("recover", "all"):
        lora = stage_recover(cfg, pruned)
    if args.stage in ("merge", "all"):
        merged = stage_merge(cfg, pruned, lora)
    if args.stage in ("fp8", "all"):
        stage_fp8(cfg, merged)

    print("\n✅ big-tier pipeline complete.")


if __name__ == "__main__":
    main()
