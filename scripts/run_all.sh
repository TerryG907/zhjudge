#!/usr/bin/env bash
# One command for the whole pipeline:
#   uv sync -> probes -> build -> validate -> train -> calibrate -> eval -> export -> compat
#
#   CONFIG=configs/train_base.yaml bash scripts/run_all.sh [--set key=value ...]
#
# Environment:
#   CONFIG          YAML config (default configs/train_base.yaml)
#   TORCH_VARIANT   auto (default) | default | cu126 | cpu — which torch 2.14.0 build to install on Linux:
#                   default = PyPI (CUDA 13.0: NVIDIA driver >= 580, compute capability >= 7.5, e.g. T4/A100/L4/4090)
#                   cu126   = CUDA 12.6 (older drivers, V100/P100)      cpu = no GPU
#   UV_EXTRAS       extra uv extras, space separated (e.g. "serve")
#   ZHJUDGE_DEVICE, ZHJUDGE_PRECISION, ZHJUDGE_THREADS, RUN_NAME, ZHJUDGE_RUNS_DIR, ZHJUDGE_DATA_DIR  (see src/zhjudge/config.py)
#   ZHJUDGE_SYNC_ONLY 1 = only install the environment (uv sync) and stop (notebooks/kaggle_train.ipynb runs the steps)
# Rerunning the same command resumes training and skips finished training.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${CONFIG:-configs/train_base.yaml}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
if [ ! -x "$UV" ]; then
  echo "uv not found. Install it with 'pip install uv' (or see https://docs.astral.sh/uv/) and rerun." >&2
  exit 1
fi

TORCH_VARIANT="${TORCH_VARIANT:-auto}"
if [ "$TORCH_VARIANT" = "auto" ]; then
  TORCH_VARIANT=default
  if [ "$(uname -s)" = "Linux" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
      drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
      cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
      if [ "${drv:-0}" -lt 580 ] 2>/dev/null || { [ -n "$cc" ] && [ "$cc" -lt 75 ] 2>/dev/null; }; then
        TORCH_VARIANT=cu126
      fi
      echo "[run_all] GPU: $(nvidia-smi -L | head -1) | driver ${drv:-?} | compute capability ${cc:-?}"
    else
      TORCH_VARIANT=cpu
    fi
  fi
fi
EXTRAS=()
case "$TORCH_VARIANT" in
  default) ;;
  cu126) EXTRAS+=(--extra cu126) ;;
  cpu) EXTRAS+=(--extra cpu) ;;
  *) echo "TORCH_VARIANT must be auto|default|cu126|cpu, got $TORCH_VARIANT" >&2; exit 2 ;;
esac
for e in ${UV_EXTRAS:-}; do EXTRAS+=(--extra "$e"); done
echo "[run_all] config $CONFIG | torch variant $TORCH_VARIANT | extras ${EXTRAS[*]:-none}"

"$UV" sync --locked ${EXTRAS[@]+"${EXTRAS[@]}"}
if [ "${ZHJUDGE_SYNC_ONLY:-0}" = "1" ]; then exit 0; fi
"$UV" run --no-sync zhjudge all --config "$CONFIG" "$@"
