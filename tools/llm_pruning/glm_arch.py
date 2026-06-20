#!/usr/bin/env python3
"""
Architecture-specific module patterns for popular LLM families.

Used by small_tier.py and big_tier.py to pick LoRA target modules and
SparseGPT / Wanda / FP8 ignore patterns from a single `arch_preset` config
field — no per-script hardcoding required.

Supported presets:
    "llama"    : Llama / Qwen / Mistral / GLM-4 dense — separate Q/K/V/O + gate/up/down.
    "mla_moe"  : DeepSeek V3 / GLM-5.2 (zai-org/GlmMoeDsaForCausalLM) —
                 MLA attention (low-rank Q/KV compressions) + MoE FFN with
                 routed + shared experts per non-dense layer.
"""

from __future__ import annotations
from typing import Dict, List


_ARCHS: Dict[str, Dict[str, List[str]]] = {
    "llama": {
        "lora_targets": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "prune_ignore": ["lm_head"],
    },
    "mla_moe": {
        # GLM-5.2 MLA + MoE structure:
        #   Attention:  q_a_proj  -> q_a_layernorm -> q_b_proj  (Q low-rank then expand)
        #               kv_a_proj_with_mqa -> kv_a_layernorm -> kv_b_proj  (KV low-rank then expand)
        #               o_proj
        #   FFN dense layers (first_k_dense_replace, e.g. first 3): gate/up/down
        #   FFN MoE layers (remaining): mlp.gate (router) + experts.N.{gate,up,down}_proj
        #                               + shared_experts.{gate,up,down}_proj
        # LoRA targets the EXPANSION projections (q_b / kv_b) and o_proj — the
        # low-rank latents (q_a / kv_a) are already compressed; further LoRA-ing
        # them is destructive. gate/up/down match both dense MLP and per-expert
        # MLP by name suffix, so all experts get adapters automatically.
        "lora_targets": [
            "q_b_proj", "kv_b_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "prune_ignore": [
            "lm_head",
            # MoE router weights — tiny, critical for routing fidelity
            "re:.*\\.mlp\\.gate$",
            "re:.*router.*",
            # MLA latent compressions — already low-rank, do not prune further
            "re:.*q_a_proj",
            "re:.*kv_a_proj_with_mqa",
            "re:.*q_a_layernorm",
            "re:.*kv_a_layernorm",
        ],
    },
}


def get_arch_preset(name: str) -> Dict[str, List[str]]:
    if name not in _ARCHS:
        raise ValueError(
            f"unknown arch_preset {name!r}; available: {list(_ARCHS)}"
        )
    return _ARCHS[name]


def resolve_lora_targets(preset: str, override: list | None) -> List[str]:
    """Return user override if non-empty, else preset's defaults."""
    if override:
        return list(override)
    return list(get_arch_preset(preset)["lora_targets"])


def resolve_prune_ignore(
    preset: str,
    extra: list | None,
    moe: bool,
    moe_ignore_router: bool,
) -> List[str]:
    """Merge preset ignores with extras; strip router ignores if MoE disabled."""
    out: List[str] = list(get_arch_preset(preset)["prune_ignore"])
    if extra:
        out += list(extra)
    if not moe or not moe_ignore_router:
        out = [p for p in out if "router" not in p and "mlp\\.gate$" not in p]
    seen: set = set()
    deduped: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped
