# LLM Pruning Pipeline — Two-Tier GLM

Production recipe for pruning + LoRA recovery + quantization of GLM models
across two heterogeneous GPU boxes, with multi-domain dataset mixing so
the specialized model retains the engineering knowledge the base already has.

All scripts are pure-Python drivers around `llm-compressor` / `peft` / `trl`
/ `AutoAWQ` / `vllm`. No agent runtime needed. Run them on your own GPU
hardware — this directory only contains code + configs.

---

## Hardware tiers

| Tier | Hardware | Compute cap | Final format | Final size (32B/70B class) |
|---|---|---|---|---|
| **Big** | RTX Pro 6000 Blackwell, 96 GB | sm_100 | **FP8 dynamic** + 2:4 sparse | ~35–40 GB |
| **Small** | 2× 2080 Ti 22 GB modded, NVLink | sm_75 | **AWQ INT4** | ~17 GB |

**Both tiers' pruning + LoRA recovery run on the Blackwell box.** Pruning on
Turing is impractical (no BF16, no 2:4 accel, slow bnb). The small tier's
final AWQ artifact ships to the 2080 Ti box for serving only.

```
Blackwell 96 GB ──► prune + recover + quant ──► SERVE big tier (FP8 + 2:4)
                              │
                              └─► ship AWQ-INT4 ──► 2× 2080 Ti SERVE small tier (TP=2)
```

---

## Layout

```
tools/llm_pruning/
├── big_tier.py                          # 4-stage driver for Blackwell big tier
├── small_tier.py                        # 4-stage driver for Turing small tier
├── glm_arch.py                          # arch_preset registry (llama / mla_moe)
├── requirements.txt                     # shared Python deps
└── configs/
    ├── big_tier.yaml                    # GLM-5.2 (MLA + MoE)
    ├── small_tier.yaml                  # GLM-4-9B-Chat fallback (Llama-style)
    ├── small_tier_coding.yaml           # coding-only specialization
    └── small_tier_eng_coding.yaml       # coding + EE/DSP/FPGA knowledge retention
```

---

## Quick start

On the **Blackwell box**:

```bash
git pull origin claude/wonderful-carson-c11wex
cd tools/llm_pruning
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Pick a preset and run all four stages:

```bash
# Big tier — GLM-5.2 753B MoE on Blackwell (see size warning below)
python big_tier.py --config configs/big_tier.yaml --stage all

# Small tier — generic small dense model
python small_tier.py --config configs/small_tier.yaml --stage all

# Small tier — coding specialist (Magicoder only)
python small_tier.py --config configs/small_tier_coding.yaml --stage all

# Small tier — coding + EE/DSP/FPGA/ASIC knowledge retention (recommended)
python small_tier.py --config configs/small_tier_eng_coding.yaml --stage all
```

Resume from any stage:

```bash
python small_tier.py --stage awq --merged-path ./out/small_tier_eng_coding/merged_fp16
```

---

## Pipeline stages

### Big tier (`big_tier.py`)
1. **sparsegpt** — SparseGPT 2:4 semi-structured pruning, per-layer CPU offload
2. **recover** — QLoRA recovery FT on the sparse base (NF4 base + BF16 LoRA)
3. **merge** — merge LoRA, re-apply 2:4 mask (dense LoRA delta breaks strict 2:4)
4. **fp8** — FP8 dynamic quantization, composes with 2:4

### Small tier (`small_tier.py`)
1. **wanda** — Wanda 50% unstructured pruning (no 2:4 — Turing has no kernel accel)
2. **recover** — QLoRA recovery FT on the sparse base
3. **merge** — merge LoRA, cast BF16 → FP16 (Turing has no BF16)
4. **awq** — AWQ INT4 quantization (group=128, zero-point)

---

## Preset comparison

| YAML | Tier | Base model | Domain | When to use |
|---|---|---|---|---|
| `big_tier.yaml` | Big | `zai-org/GLM-5.2-FP8` | general | If you have the hardware for 753B |
| `small_tier.yaml` | Small | `zai-org/GLM-4-9B-Chat` | general | Baseline / non-coding use case |
| `small_tier_coding.yaml` | Small | `zai-org/GLM-4-9B-Chat` | coding only | Pure code assistant |
| `small_tier_eng_coding.yaml` | Small | `zai-org/GLM-4-9B-Chat` | coding + EE/DSP/FPGA/ASIC | **Recommended for engineering work** |

---

## CLI overrides (both scripts)

Override any preset without editing YAML:

| Flag | Purpose |
|---|---|
| `--model-id` | Swap base model (e.g. `Qwen/Qwen2.5-Coder-32B-Instruct`) |
| `--recover-dataset` | Single-source override (e.g. `ise-uiuc/Magicoder-Evol-Instruct-110K`) |
| `--recover-samples` | Total samples in recovery FT |
| `--lora-epochs` | More epochs = stronger memorization (try 2–3 for domains) |
| `--output-root` | Per-domain output dir for isolated runs |
| `--stage` | `wanda`/`sparsegpt`/`recover`/`merge`/`awq`/`fp8`/`all` |
| `--pruned-path` / `--lora-path` / `--merged-path` | Resume from a prior stage's output |

---

## Multi-domain dataset mixing — preserve what the base already knows

**The problem.** Pruning damages knowledge uniformly. If you recover with only
Magicoder, you get a coding expert that forgot DSP / wireless / RF math. The
fix: mix code instruct + general SFT + STEM + your domain docs during recovery.

**The mechanism.** Both scripts accept a `recover_datasets:` list in YAML where
each entry is `{name, weight, split?, subset?}`. Weights are fractions of
`recover_samples`; they don't need to sum to 1.0. Local `.jsonl` paths are
auto-detected. Schema is normalized via `_to_chat_text` so `messages`,
`instruction`/`output`, `prompt`/`response`, `question`/`answer` all work.

**Example** (from `small_tier_eng_coding.yaml`):

```yaml
recover_samples: 100000
recover_datasets:
  - { name: "ise-uiuc/Magicoder-Evol-Instruct-110K", weight: 0.35 }  # 35k coding
  - { name: "ise-uiuc/Magicoder-OSS-Instruct-75K",   weight: 0.10 }  # 10k OSS code
  - { name: "garage-bAInd/Open-Platypus",            weight: 0.15 }  # 15k STEM
  - { name: "HuggingFaceH4/ultrachat_200k",  split: "train_sft", weight: 0.20 }  # general
  - { name: "Open-Orca/OpenOrca",                    weight: 0.10 }
  - { name: "shailja/Verilog_GitHub",                weight: 0.05 }
  # - { name: "./data/my_wireless_dsp_qa.jsonl",     weight: 0.05 }  # YOURS
```

### Bring-your-own-domain jsonl

Public datasets for wireless / RF / DSP / FPGA / ASIC are scarce. Bring your
own textbooks, app notes, datasheets, internal docs as Q&A pairs:

```jsonl
{"instruction": "Derive the noise figure of a cascaded LNA-mixer.", "output": "Using Friis' formula: F_total = F1 + (F2-1)/G1 + ..."}
{"instruction": "Why pipeline a CIC decimator in fixed-point HDL?", "output": "Each stage's bit growth ..."}
{"messages":[{"role":"user","content":"Explain TDD slot reconfiguration penalty in 5G NR"},{"role":"assistant","content":"..."}]}
```

Even **1k–5k high-quality samples per niche** moves the needle dramatically.
Quality beats quantity for sparse domains.

---

## arch_preset — module-name dispatch

`glm_arch.py` provides two presets driving LoRA target modules and prune ignores:

| Preset | LoRA targets | Use for |
|---|---|---|
| `llama` | `q/k/v/o_proj`, `gate/up/down_proj` | Llama, Qwen, Mistral, GLM-4 dense |
| `mla_moe` | `q_b_proj`, `kv_b_proj`, `o_proj`, `gate/up/down_proj` (matches every routed/shared expert by suffix) | GLM-5.2 (`GlmMoeDsaForCausalLM`), DeepSeek V3 |

`mla_moe` also adds prune-ignore patterns for MLA latent projections
(`q_a_proj`, `kv_a_proj_with_mqa` — already low-rank, do not prune further)
and MoE router weights (`re:.*\.mlp\.gate$`, tiny + critical for routing).

Pick via `arch_preset:` in YAML. Leave `lora_target_modules: []` to use the
preset's defaults; non-empty list overrides them.

---

## Quantization formats — why FP8 vs AWQ INT4 per tier

| Tier | Hardware | Format | Why |
|---|---|---|---|
| Big | Blackwell sm_100 | FP8 dynamic (`FP8_DYNAMIC`) | Native FP8 tensor cores, composes with 2:4 |
| Small | Turing sm_75 | AWQ INT4 (`awq_bits: 4`) | No FP8 hardware on Turing; AWQ has working kernels via ExLlamaV2 / vLLM |

INT8 is technically possible on Turing but worse: 32B in INT8 ≈ 32 GB, barely
fits TP=2 on 2×22 GB before KV cache eats the rest. INT4 (~17 GB) leaves
headroom for context. If you specifically need INT8 quality (e.g. for 7–9B),
set `awq_bits: 8`.

---

## Serving

After the pipelines finish, each script prints the exact `vllm serve` command.

**Big tier (Blackwell):**
```bash
vllm serve out/big_tier/fp8_24 \
  --dtype auto --kv-cache-dtype fp8 \
  --max-model-len 32768 --enable-prefix-caching --max-num-seqs 256
```

**Small tier (2× 2080 Ti, after rsync):**
```bash
vllm serve /models/glm5-32b-awq \
  --dtype float16 --quantization awq \
  --tensor-parallel-size 2 \
  --max-model-len 8192 --enable-prefix-caching --max-num-seqs 64
```

Pin **CUDA 12.1 + PyTorch 2.3.x** on the 2080 Ti box — newer vLLM/FA2 wheels
drop sm_75. Fall back to TGI or ExLlamaV2 + tabbyAPI if vLLM kernels misbehave.

---

## GLM-5.2 reality check

`zai-org/GLM-5.2` is 753B MoE. Even FP8 weights are ~376 GB — won't fit in a
single 96 GB GPU under any format. The big-tier script uses llm-compressor's
per-layer CPU/disk offload so the recipe still executes on one Blackwell, but
expect **days, not hours**, and need TBs of NVMe at `offload_folder`. For
practical timelines you want a multi-GPU node (4× 96 GB minimum) or cloud.

Architecture support is correct; the memory wall is a separate problem.

---

## Acceptance criteria

| Tier | Throughput | Quality | Notes |
|---|---|---|---|
| Big | ≥2.5× tokens/s vs BF16 baseline | ≤2 pts MMLU drop, ≤3 pts GSM8K | Serves at 32k context |
| Small | ≥30 tok/s/user @ concurrency 16 | ≤3 pts MMLU drop, ≤5 pts GSM8K | Serves on 2× 2080 Ti TP=2 |

Evaluate with `lm_eval --model vllm` and `vllm bench serve` after each tier's
final stage.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GLM-5.2 wrong module names in LoRA targets` | Set `arch_preset: "mla_moe"` |
| `AutoAWQ unsupported architecture: glm_moe_dsa` | Don't run small tier on GLM-5.2 — use a Llama-style base |
| `vLLM crash on sm_75 (Turing)` | Pin `torch==2.3.1` + CUDA 12.1; or use TGI / ExLlamaV2 |
| `Modded 2080 Ti thermal throttle` | `nvidia-smi -l 1`, drop `--max-num-seqs`, ensure airflow |
| `70B doesn't fit at 32k context` | Add `--kv-cache-dtype fp8` (already on); drop to 16k |
| `LoRA recovery insufficient at 50% 2:4` | Bump `lora_rank` to 128–256; add 1 epoch; fall back to 40% sparse |
| `Chinese eval drops more than English` | Add ShareGPT-zh or CLUECorpus to `recover_datasets` |
| `Domain knowledge lost after specialization` | Use `small_tier_eng_coding.yaml` and add your own domain jsonl |
