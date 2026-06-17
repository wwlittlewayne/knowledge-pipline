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
from typing import List

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.pruning import WandaPruningModifier

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from awq import AutoAWQForCausalLM


@dataclass
class Config:
    model_id: str = "THUDM/glm-5-32b"
    output_root: Path = Path("./out/small_tier")

    sparsity: float = 0.5
    calib_dataset: str = "wikitext"
    calib_subset: str = "wikitext-2-raw-v1"
    calib_samples: int = 512
    calib_seqlen: int = 2048

    recover_dataset: str = "Open-Orca/OpenOrca"
    recover_samples: int = 50_000
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_epochs: int = 1
    lora_lr: float = 1e-4
    lora_batch: int = 1
    lora_grad_accum: int = 8
    lora_seqlen: int = 2048
    use_qlora_base: bool = True

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
        target_modules=cfg.lora_target_modules,
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
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
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
