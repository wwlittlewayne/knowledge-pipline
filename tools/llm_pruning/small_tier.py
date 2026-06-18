#!/usr/bin/env python3
"""
Small-tier GLM-5 prune + recover + quantize pipeline.

Target hardware (for the FINAL artifact):
    2x RTX 2080 Ti 22GB NVLink (Turing sm_75) — FP16 only, no BF16, no FA2,
    no 2:4 sparse acceleration. We use unstructured Wanda (memory shape only)
    + AWQ INT4 to fit a 32B-class GLM-5 with TP=2.

Run THIS pipeline on the 96GB RTX Pro 6000 (Blackwell) box — pruning + LoRA
recovery need more VRAM than the 2080 Tis have. The final AWQ artifact is
~17GB and gets rsynced to the Turing serving box.

Stages (each saves to disk; resume from any stage via --stage / --*-path):
    1. wanda    : 50% unstructured Wanda pruning on calibration data
    2. recover  : QLoRA recovery fine-tune on the pruned base (~50k samples)
    3. merge    : merge LoRA adapter into pruned base, cast to FP16
    4. awq      : AWQ INT4 quantization, ready for vLLM TP=2 on Turing

Usage:
    # Full pipeline
    python small_tier.py --config configs/small_tier.yaml --stage all

    # Single stage, resuming from prior outputs
    python small_tier.py --stage recover --pruned-path ./out/small_tier/wanda
    python small_tier.py --stage awq     --merged-path ./out/small_tier/merged_fp16
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.pruning import WandaPruningModifier

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from awq import AutoAWQForCausalLM

from glm_arch import resolve_lora_targets, resolve_prune_ignore


@dataclass
class Config:
    # NOTE: GLM-5.2 only ships at 753B MoE — no small variant exists.
    # Small tier defaults to a smaller dense GLM that fits TP=2 on 2x 2080 Ti.
    # Override model_id + arch_preset in YAML for your actual small model.
    model_id: str = "zai-org/GLM-4-9B-Chat"
    output_root: Path = Path("./out/small_tier")

    # "llama" for dense GLM-4 / Llama / Qwen; "mla_moe" for GLM-5.2 (AWQ may
    # not support glm_moe_dsa — small tier targets a dense model in practice).
    arch_preset: str = "llama"

    sparsity: float = 0.5
    calib_dataset: str = "wikitext"
    calib_subset: str = "wikitext-2-raw-v1"
    calib_samples: int = 512
    calib_seqlen: int = 2048

    recover_dataset: str = "Open-Orca/OpenOrca"
    # Multi-domain mixing: list of {name, weight, split?} dicts. When non-empty
    # this OVERRIDES recover_dataset. weight is the share of recover_samples
    # drawn from that source (need not sum to 1.0). Use local .jsonl paths for
    # proprietary domain data (wireless, DSP, FPGA, ASIC textbooks/specs).
    recover_datasets: List[Dict] = field(default_factory=list)
    recover_samples: int = 50_000
    lora_rank: int = 64
    lora_alpha: int = 128
    # Empty → use arch_preset's lora_targets. Non-empty list overrides.
    lora_target_modules: List[str] = field(default_factory=list)
    lora_epochs: int = 1
    lora_lr: float = 1e-4
    lora_batch: int = 1
    lora_grad_accum: int = 8
    lora_seqlen: int = 2048
    use_qlora_base: bool = True

    # MoE knobs (router weight handling) — usually false for the small tier
    moe: bool = False
    moe_ignore_router: bool = True

    awq_bits: int = 4
    awq_group_size: int = 128
    awq_zero_point: bool = True
    awq_calib_samples: int = 128

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if "output_root" in data:
            data["output_root"] = Path(data["output_root"])
        return cls(**data)


def _load_calib_texts(cfg: Config, n: int) -> list[str]:
    raw = load_dataset(cfg.calib_dataset, cfg.calib_subset, split="train")
    texts = [r["text"] for r in raw if r.get("text") and len(r["text"]) > 200]
    return texts[:n]


def _to_chat_text(ex, tokenizer):
    """Normalize one example to a chat-templated string regardless of schema."""
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


def _load_one_dataset(spec: Dict, n_target: int):
    """Load one HF dataset id or local jsonl, shuffled, capped at n_target."""
    name = spec["name"]
    split = spec.get("split", "train")
    subset = spec.get("subset")

    if name.endswith(".jsonl") or name.endswith(".json"):
        ds = load_dataset("json", data_files=name, split="train")
    elif subset:
        ds = load_dataset(name, subset, split=split)
    else:
        ds = load_dataset(name, split=split)

    ds = ds.shuffle(seed=42)
    return ds.select(range(min(n_target, len(ds))))


def _load_recovery_data(cfg: Config, tokenizer):
    """Load and concat all recovery datasets, normalized to {text: ...}."""
    specs = cfg.recover_datasets if cfg.recover_datasets else [
        {"name": cfg.recover_dataset, "weight": 1.0}
    ]

    parts = []
    for spec in specs:
        weight = spec.get("weight", 1.0)
        n_target = max(1, int(cfg.recover_samples * weight))
        ds = _load_one_dataset(spec, n_target)
        ds = ds.map(
            lambda ex: _to_chat_text(ex, tokenizer),
            remove_columns=ds.column_names,
        )
        print(f"[recover] + {spec['name']}: {len(ds)} samples (weight={weight})")
        parts.append(ds)

    combined = concatenate_datasets(parts).shuffle(seed=42)
    print(f"[recover] total mixed dataset: {len(combined)} samples")
    return combined


def stage_wanda(cfg: Config) -> Path:
    out = cfg.output_root / "wanda"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[wanda] model={cfg.model_id} target_sparsity={cfg.sparsity:.0%}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    raw = load_dataset(cfg.calib_dataset, cfg.calib_subset, split="train")

    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=cfg.calib_seqlen)

    calib = (
        raw.filter(lambda r: r.get("text") and len(r["text"]) > 200)
           .select(range(min(cfg.calib_samples * 2, len(raw))))
           .map(tok, remove_columns=raw.column_names)
    )
    calib = calib.filter(lambda x: len(x["input_ids"]) >= cfg.calib_seqlen // 2)
    calib = calib.select(range(min(cfg.calib_samples, len(calib))))

    recipe = WandaPruningModifier(
        sparsity=cfg.sparsity,
        mask_structure="unstructured",
        targets=["Linear"],
        ignore=resolve_prune_ignore(
            cfg.arch_preset, None, cfg.moe, cfg.moe_ignore_router,
        ),
    )

    oneshot(
        model=cfg.model_id,
        dataset=calib,
        recipe=recipe,
        max_seq_length=cfg.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(out),
        trust_remote_code_model=True,
    )
    tokenizer.save_pretrained(out)
    print(f"[wanda] saved → {out}")
    return out


def stage_recover(cfg: Config, pruned_path: Path) -> Path:
    out = cfg.output_root / "lora"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[recover] base={pruned_path} qlora_base={cfg.use_qlora_base}")

    if cfg.use_qlora_base:
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
    else:
        base = AutoModelForCausalLM.from_pretrained(
            pruned_path,
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

    ds = _load_recovery_data(cfg, tokenizer)

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
    out = cfg.output_root / "merged_fp16"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[merge] base={pruned_path} adapter={lora_path}")

    base = AutoModelForCausalLM.from_pretrained(
        pruned_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, lora_path).merge_and_unload()
    merged = merged.to(torch.float16)

    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(pruned_path, trust_remote_code=True).save_pretrained(out)
    print(f"[merge] FP16 merged model → {out}")
    return out


def stage_awq(cfg: Config, merged_path: Path) -> Path:
    out = cfg.output_root / "awq_int4"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[awq] model={merged_path} bits={cfg.awq_bits} group={cfg.awq_group_size}")

    model = AutoAWQForCausalLM.from_pretrained(
        str(merged_path),
        safetensors=True,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True)

    quant_config = {
        "zero_point": cfg.awq_zero_point,
        "q_group_size": cfg.awq_group_size,
        "w_bit": cfg.awq_bits,
        "version": "GEMM",
    }

    calib_texts = _load_calib_texts(cfg, cfg.awq_calib_samples)
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(out)
    print(f"[awq] saved → {out}")
    print(f"[awq] ship to 2080 Ti serving box:")
    print(f"      rsync -av --progress {out}/ user@turing-box:/models/glm5-32b-awq/")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config; overrides defaults")
    p.add_argument("--stage", choices=["wanda", "recover", "merge", "awq", "all"],
                   default="all")
    p.add_argument("--pruned-path", type=Path)
    p.add_argument("--lora-path", type=Path)
    p.add_argument("--merged-path", type=Path)

    # Domain / dataset overrides — apply on top of YAML defaults.
    p.add_argument("--model-id",
                   help="HF model id, overrides Config.model_id")
    p.add_argument("--recover-dataset",
                   help="HF dataset for LoRA recovery (e.g. ise-uiuc/Magicoder-Evol-Instruct-110K for coding)")
    p.add_argument("--recover-samples", type=int,
                   help="Number of recovery samples to take")
    p.add_argument("--lora-epochs", type=int,
                   help="LoRA recovery epochs (try 2-3 for stronger domain specialization)")
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

    pruned = args.pruned_path or (cfg.output_root / "wanda")
    lora = args.lora_path or (cfg.output_root / "lora")
    merged = args.merged_path or (cfg.output_root / "merged_fp16")

    if args.stage in ("wanda", "all"):
        pruned = stage_wanda(cfg)
    if args.stage in ("recover", "all"):
        lora = stage_recover(cfg, pruned)
    if args.stage in ("merge", "all"):
        merged = stage_merge(cfg, pruned, lora)
    if args.stage in ("awq", "all"):
        stage_awq(cfg, merged)

    print("\n✅ small-tier pipeline complete.")


if __name__ == "__main__":
    main()
