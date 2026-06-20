#!/usr/bin/env python3
"""
Mid-tier prune + recover + quantize pipeline.

Target hardware: 2x RTX 3090 24GB NVLink + 2x EPYC 75F3 + 256GB DDR4
                 Ampere sm_86: BF16, 2:4 sparse tensor cores, FA2,
                 Marlin INT4 kernels (W4A16).

Two engines supported, llama.cpp is the default because EPYC 75F3 +
DDR4-3200 8-channel + AVX-512 actually COMPUTES the CPU-resident MoE
experts (~1.5x single-user throughput vs vLLM offload). Pick vLLM only
when serving 4+ concurrent users.

Stages dispatched by engine:
    engine="llama_cpp" (default)  →  recover → merge → gguf
    engine="vllm"                 →  sparsegpt → recover → merge → w4a16

Usage:
    python mid_tier.py --config configs/mid_tier_max.yaml --stage all
    python mid_tier.py --config configs/mid_tier_70b.yaml --stage all
    python mid_tier.py --engine vllm --stage all   # override default
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.obcq import SparseGPTModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

from glm_arch import resolve_lora_targets, resolve_prune_ignore


@dataclass
class Config:
    model_id: str = "deepseek-ai/DeepSeek-V2.5"
    output_root: Path = Path("./out/mid_tier")
    offload_folder: Path = Path("/tmp/llmcompressor_offload")

    # "llama_cpp" (default) or "vllm"
    engine: str = "llama_cpp"

    # "mla_moe" for DeepSeek V2/V3 / GLM-5.2; "llama" for dense Llama / Qwen / GLM-4
    arch_preset: str = "mla_moe"

    # SparseGPT (vLLM path only — GGUF has no 2:4 support)
    sparsity: float = 0.5
    mask_structure: str = "2:4"

    calib_dataset: str = "wikitext"
    calib_subset: str = "wikitext-2-raw-v1"
    calib_samples: int = 512
    calib_seqlen: int = 2048

    recover_dataset: str = "Open-Orca/OpenOrca"
    recover_datasets: List[Dict] = field(default_factory=list)
    recover_samples: int = 50_000
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_target_modules: List[str] = field(default_factory=list)
    lora_epochs: int = 1
    lora_lr: float = 1e-4
    lora_batch: int = 1
    lora_grad_accum: int = 16
    lora_seqlen: int = 2048
    use_qlora_base: bool = True

    moe: bool = True
    moe_ignore_router: bool = True

    # vLLM W4A16 stage
    w4a16_scheme: str = "W4A16"
    w4a16_ignore: List[str] = field(default_factory=list)
    remask_calib_samples: int = 128

    # llama.cpp GGUF stage
    llama_cpp_dir: Path = Path("/opt/llama.cpp")
    gguf_quant: str = "Q4_K_M"

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for key in ("output_root", "offload_folder", "llama_cpp_dir"):
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
        cfg.arch_preset, cfg.w4a16_ignore, cfg.moe, cfg.moe_ignore_router,
    )


def _to_chat_text(ex, tokenizer):
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


def stage_sparsegpt(cfg: Config) -> Path:
    """SparseGPT 2:4 — vLLM path only; GGUF has no 2:4 support."""
    if cfg.engine == "llama_cpp":
        print("[sparsegpt] SKIP — engine=llama_cpp; GGUF has no 2:4 support")
        return Path()

    out = cfg.output_root / "sparsegpt"
    out.mkdir(parents=True, exist_ok=True)
    cfg.offload_folder.mkdir(parents=True, exist_ok=True)
    print(f"[sparsegpt] model={cfg.model_id} mask={cfg.mask_structure} "
          f"sparsity={cfg.sparsity:.0%}")

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


def stage_recover(cfg: Config, base_path: Path) -> Path:
    """QLoRA recovery. For llama.cpp engine, base_path is the raw HF model_id."""
    out = cfg.output_root / "lora"
    out.mkdir(parents=True, exist_ok=True)

    base_arg = str(base_path) if base_path and base_path.exists() else cfg.model_id
    print(f"[recover] base={base_arg} qlora_base={cfg.use_qlora_base}")

    if cfg.use_qlora_base:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_arg, quantization_config=bnb, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_arg, device_map="auto",
            trust_remote_code=True, torch_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_arg, trust_remote_code=True)
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
        model=base, args=sft_args, train_dataset=ds,
        peft_config=lora_cfg, tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out))
    print(f"[recover] LoRA adapter saved → {out}")
    return out


def stage_merge(cfg: Config, base_path: Path, lora_path: Path) -> Path:
    """Merge LoRA into base. For vLLM engine, re-apply 2:4 mask after merge."""
    out = cfg.output_root / "merged_bf16"
    out.mkdir(parents=True, exist_ok=True)

    base_arg = str(base_path) if base_path and base_path.exists() else cfg.model_id
    print(f"[merge] base={base_arg} adapter={lora_path}")

    base = AutoModelForCausalLM.from_pretrained(
        base_arg, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, offload_folder=str(cfg.offload_folder),
    )
    merged = PeftModel.from_pretrained(base, lora_path).merge_and_unload()

    pre_remask = out / "pre_remask" if cfg.engine == "vllm" else out
    pre_remask.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(pre_remask), safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(base_arg, trust_remote_code=True).save_pretrained(pre_remask)
    del base, merged
    torch.cuda.empty_cache()

    if cfg.engine == "llama_cpp":
        print(f"[merge] saved (no re-mask, llama_cpp engine) → {out}")
        return out

    print(f"[merge] re-applying {cfg.mask_structure} mask via SparseGPT "
          f"({cfg.remask_calib_samples} calib samples)")
    calib, _ = _build_calib(cfg, cfg.remask_calib_samples)
    recipe = SparseGPTModifier(
        sparsity=cfg.sparsity, mask_structure=cfg.mask_structure,
        targets=["Linear"], ignore=_ignore_targets(cfg),
    )
    oneshot(
        model=str(pre_remask), dataset=calib, recipe=recipe,
        max_seq_length=cfg.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(out), trust_remote_code_model=True,
        oneshot_device="auto", offload_folder=str(cfg.offload_folder),
    )
    AutoTokenizer.from_pretrained(base_arg, trust_remote_code=True).save_pretrained(out)
    print(f"[merge] merged + re-masked → {out}")
    return out


def stage_w4a16(cfg: Config, merged_path: Path) -> Path:
    """W4A16 INT4 quant via llm-compressor (vLLM/Marlin path)."""
    out = cfg.output_root / "w4a16"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[w4a16] model={merged_path} scheme={cfg.w4a16_scheme}")

    calib, _ = _build_calib(cfg, cfg.calib_samples // 2)
    recipe = QuantizationModifier(
        targets="Linear", scheme=cfg.w4a16_scheme,
        ignore=_ignore_targets(cfg),
    )
    oneshot(
        model=str(merged_path), dataset=calib, recipe=recipe,
        max_seq_length=cfg.calib_seqlen,
        num_calibration_samples=len(calib),
        output_dir=str(out), trust_remote_code_model=True,
        oneshot_device="auto", offload_folder=str(cfg.offload_folder),
    )
    AutoTokenizer.from_pretrained(merged_path, trust_remote_code=True).save_pretrained(out)
    print(f"[w4a16] saved → {out}")
    print()
    print(f"[w4a16] serve on 2× 3090 with vLLM:")
    print(f"      vllm serve {out} \\")
    print(f"        --tensor-parallel-size 2 --dtype bfloat16 \\")
    print(f"        --cpu-offload-gb 85 \\")
    print(f"        --max-model-len 16384 \\")
    print(f"        --enable-prefix-caching --max-num-seqs 16 \\")
    print(f"        --enforce-eager")
    return out


def stage_gguf(cfg: Config, merged_path: Path) -> Path:
    """Convert merged HF → GGUF f16 → quantize to K-quant (llama.cpp path)."""
    out = cfg.output_root / "gguf"
    out.mkdir(parents=True, exist_ok=True)

    convert_script = cfg.llama_cpp_dir / "convert_hf_to_gguf.py"
    quantize_bin = cfg.llama_cpp_dir / "build" / "bin" / "llama-quantize"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}. "
            f"Set llama_cpp_dir in the YAML to your llama.cpp install."
        )
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize not found at {quantize_bin}. "
            f"Build llama.cpp first: cmake -B build && cmake --build build -j"
        )

    name = merged_path.name
    f16_gguf = out / f"{name}-f16.gguf"
    final_gguf = out / f"{name}-{cfg.gguf_quant.lower()}.gguf"

    print(f"[gguf] HF → GGUF f16: {merged_path} → {f16_gguf}")
    subprocess.run(
        ["python", str(convert_script), str(merged_path),
         "--outfile", str(f16_gguf), "--outtype", "f16"],
        check=True,
    )

    print(f"[gguf] quantize f16 → {cfg.gguf_quant}: {final_gguf}")
    subprocess.run(
        [str(quantize_bin), str(f16_gguf), str(final_gguf), cfg.gguf_quant],
        check=True,
    )

    if f16_gguf.exists():
        f16_gguf.unlink()
        print(f"[gguf] cleaned intermediate {f16_gguf.name}")

    print(f"[gguf] final artifact → {final_gguf}")
    _print_serve_commands(cfg, final_gguf)
    return final_gguf


def _print_serve_commands(cfg: Config, gguf: Path) -> None:
    """Emit arch-aware llama-server recipes for both the 2x 3090 + EPYC box
    and the 2x 2080 Ti box (cross-tier serving)."""
    print()
    print(f"[gguf] ship to your serving box:")
    print(f"      rsync -av --progress {gguf} user@<host>:/models/")
    print()

    if cfg.moe:
        # MoE: --n-cpu-moe controls how many EXPERT layers stay on CPU.
        print(f"[serve] 2× 3090 24GB + 2× EPYC 75F3 (recommended for MoE):")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print(f"        -ngl 999 --n-cpu-moe 32 \\           # tune up to free more VRAM for KV")
        print(f"        -c 16384 --parallel 1 \\")
        print(f"        --threads 64 --numa distribute --mlock \\")
        print(f"        --host 0.0.0.0 --port 8080")
        print()
        print(f"[serve] 2× 2080 Ti 22GB (less VRAM, no FA2 — push more experts to CPU):")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print(f"        -ngl 999 --n-cpu-moe 56 \\           # more on CPU due to smaller VRAM")
        print(f"        -c 8192 --parallel 1 \\              # smaller ctx — KV pressure worse w/o FA2")
        print(f"        --threads $(nproc) --mlock \\")
        print(f"        --host 0.0.0.0 --port 8080")
    else:
        # Dense: --n-cpu-moe is a no-op. Use -ngl <N> for partial offload.
        print(f"[serve] 2× 3090 24GB + 2× EPYC 75F3 (fits VRAM, no offload):")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print(f"        -ngl 999 --split-mode row \\         # all layers on GPU, TP=2")
        print(f"        -c 32768 --parallel 1 \\")
        print(f"        --threads 64 --numa distribute --mlock \\")
        print(f"        --host 0.0.0.0 --port 8080")
        print()
        print(f"[serve] 2× 2080 Ti 22GB (44GB total, dense weights):")
        print(f"      # 30B Q5_K_M (~22GB)  → fits one card; consider --tensor-split 1,0")
        print(f"      # 70B Q4_K_M (~40GB)  → fits TP=2 tight; for KV headroom rebuild as Q3_K_M")
        print(f"      # If OOM: drop -ngl to e.g. 60 to push later layers to CPU")
        print(f"      llama-server -m /models/{gguf.name} \\")
        print(f"        -ngl 999 --split-mode row \\")
        print(f"        -c 8192 --parallel 1 \\")
        print(f"        --threads $(nproc) --mlock \\")
        print(f"        --host 0.0.0.0 --port 8080")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--stage",
                   choices=["sparsegpt", "recover", "merge", "w4a16", "gguf", "all"],
                   default="all")
    p.add_argument("--engine", choices=["llama_cpp", "vllm"],
                   help="Override Config.engine (default: llama_cpp)")
    p.add_argument("--pruned-path", type=Path)
    p.add_argument("--lora-path", type=Path)
    p.add_argument("--merged-path", type=Path)

    p.add_argument("--model-id")
    p.add_argument("--recover-dataset")
    p.add_argument("--recover-samples", type=int)
    p.add_argument("--lora-epochs", type=int)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--gguf-quant",
                   help="Override GGUF target (Q4_K_M, Q5_K_M, Q6_K, Q8_0, ...)")
    p.add_argument("--llama-cpp-dir", type=Path,
                   help="Path to llama.cpp install (must contain convert_hf_to_gguf.py and build/bin/llama-quantize)")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.engine:
        cfg.engine = args.engine
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
    if args.gguf_quant:
        cfg.gguf_quant = args.gguf_quant
    if args.llama_cpp_dir:
        cfg.llama_cpp_dir = args.llama_cpp_dir

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    pruned = args.pruned_path or (cfg.output_root / "sparsegpt")
    lora = args.lora_path or (cfg.output_root / "lora")
    merged = args.merged_path or (cfg.output_root / "merged_bf16")

    print(f"[mid_tier] engine={cfg.engine} model={cfg.model_id}")

    if cfg.engine == "llama_cpp":
        run_stages = {"all": ["recover", "merge", "gguf"]}.get(
            args.stage, [args.stage]
        )
    else:
        run_stages = {"all": ["sparsegpt", "recover", "merge", "w4a16"]}.get(
            args.stage, [args.stage]
        )

    if "sparsegpt" in run_stages:
        pruned = stage_sparsegpt(cfg)
    if "recover" in run_stages:
        base_for_recover = pruned if cfg.engine == "vllm" else Path()
        lora = stage_recover(cfg, base_for_recover)
    if "merge" in run_stages:
        base_for_merge = pruned if cfg.engine == "vllm" else Path()
        merged = stage_merge(cfg, base_for_merge, lora)
    if "w4a16" in run_stages:
        stage_w4a16(cfg, merged)
    if "gguf" in run_stages:
        stage_gguf(cfg, merged)

    print(f"\n✅ mid-tier pipeline complete (engine={cfg.engine}).")


if __name__ == "__main__":
    main()
