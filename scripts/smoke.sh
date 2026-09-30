#!/usr/bin/env bash
# CPU smoke test of the whole pipeline (tiny model, ~200 examples, 20 steps). Forces CPU even when a GPU/MPS exists.
#   nice -n 15 bash scripts/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export ZHJUDGE_DEVICE=cpu
export ZHJUDGE_THREADS="${ZHJUDGE_THREADS:-4}"
export CONFIG=configs/train_smoke.yaml
exec bash scripts/run_all.sh "$@"
