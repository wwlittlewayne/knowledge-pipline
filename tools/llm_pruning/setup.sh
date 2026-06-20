#!/usr/bin/env bash
#
# Auto-detecting setup for the LLM pruning + quantization toolchain.
#
# Detects the GPU compute capability via nvidia-smi, installs the matching
# CUDA PyTorch wheel, then the shared Python deps (transformers, llmcompressor,
# peft, trl, autoawq, …). Optionally clones + builds llama.cpp with CUDA for
# the GGUF quant scripts and the mid-tier llama.cpp engine.
#
# ONLINE boxes only (uses pip / git / cmake over the network).
#
# Usage:
#   ./setup.sh                              # python deps only, auto CUDA
#   ./setup.sh --venv                       # create+use ./.venv first
#   ./setup.sh --llama-cpp-dir /opt/llama.cpp   # also build llama.cpp there
#   ./setup.sh --skip-torch                 # don't touch torch (already installed)
#
# Arch → CUDA wheel mapping:
#   7.5 Turing (2080 Ti) · 8.6 Ampere (3090) · 8.9 Ada · 9.0 Hopper  → cu124
#   10.0 B200 · 12.0 RTX PRO 6000 Blackwell                          → cu128
#   (no GPU detected)                                                → cpu

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_CPP_DIR=""
MAKE_VENV=0
SKIP_TORCH=0

usage() { sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama-cpp-dir) LLAMA_CPP_DIR="${2:?path required}"; shift 2 ;;
    --venv)          MAKE_VENV=1; shift ;;
    --skip-torch)    SKIP_TORCH=1; shift ;;
    -h|--help)       usage ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "==> LLM toolchain setup (online)"

# ---- optional venv ----------------------------------------------------------
if [[ "$MAKE_VENV" == "1" ]]; then
  echo "==> creating virtualenv at $HERE/.venv"
  python3 -m venv "$HERE/.venv"
  # shellcheck disable=SC1091
  source "$HERE/.venv/bin/activate"
fi

PIP="${PIP:-pip}"
$PIP install --upgrade pip setuptools wheel >/dev/null

# ---- detect GPU compute capability -----------------------------------------
CAP=""
if command -v nvidia-smi >/dev/null 2>&1; then
  # take the first GPU; warn if a second GPU reports a different cap
  mapfile -t CAPS < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' ')
  CAP="${CAPS[0]:-}"
  if [[ ${#CAPS[@]} -gt 1 ]]; then
    for c in "${CAPS[@]}"; do
      [[ "$c" != "$CAP" ]] && echo "⚠️  mixed GPU compute caps detected (${CAPS[*]}); using $CAP" && break
    done
  fi
fi

# ---- map cap -> torch CUDA index --------------------------------------------
case "$CAP" in
  "")            IDX="https://download.pytorch.org/whl/cpu";   LABEL="CPU (no GPU detected)";;
  10.*|11.*|12.*) IDX="https://download.pytorch.org/whl/cu128"; LABEL="Blackwell/sm_$CAP → CUDA 12.8";;
  *)             IDX="https://download.pytorch.org/whl/cu124"; LABEL="sm_$CAP → CUDA 12.4";;
esac
echo "==> GPU compute_cap='${CAP:-none}'  →  torch wheel: $LABEL"

# ---- install torch ----------------------------------------------------------
if [[ "$SKIP_TORCH" == "1" ]]; then
  echo "==> --skip-torch set; leaving torch as-is"
else
  echo "==> installing torch from $IDX"
  $PIP install torch --index-url "$IDX"
fi

# ---- install the rest of the toolchain --------------------------------------
echo "==> installing Python deps from requirements.txt"
$PIP install -r "$HERE/requirements.txt"

python3 - <<'PY'
import importlib
ok, miss = [], []
for m in ("torch","transformers","datasets","llmcompressor","peft","trl","yaml"):
    try:
        importlib.import_module(m); ok.append(m)
    except Exception:
        miss.append(m)
print("   ✅ import ok :", ", ".join(ok))
if miss:
    print("   ⚠️ still missing:", ", ".join(miss))
try:
    import torch
    print(f"   torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
except Exception:
    pass
PY

# ---- optional: build llama.cpp ---------------------------------------------
if [[ -n "$LLAMA_CPP_DIR" ]]; then
  echo "==> setting up llama.cpp at $LLAMA_CPP_DIR"
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
  else
    echo "   repo exists; pulling latest"
    git -C "$LLAMA_CPP_DIR" pull --ff-only || true
  fi

  CUDA_FLAG="-DGGML_CUDA=ON"
  if [[ -z "$CAP" ]]; then
    echo "   no GPU → building CPU-only llama.cpp"
    CUDA_FLAG="-DGGML_CUDA=OFF"
  fi
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" $CUDA_FLAG
  cmake --build "$LLAMA_CPP_DIR/build" -j "$(nproc)"
  # python deps for convert_hf_to_gguf.py
  [[ -f "$LLAMA_CPP_DIR/requirements.txt" ]] && $PIP install -r "$LLAMA_CPP_DIR/requirements.txt"

  echo "   ✅ llama.cpp built. Pass to GGUF scripts with:"
  echo "        --llama-cpp-dir $LLAMA_CPP_DIR"
fi

echo "==> done."
[[ "$MAKE_VENV" == "1" ]] && echo "   (activate later with: source $HERE/.venv/bin/activate)"
