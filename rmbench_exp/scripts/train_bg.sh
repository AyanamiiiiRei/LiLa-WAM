#!/bin/bash
# Start (or resume) training in the background, like the nohup launches used for RoboTwin.
# Closing the terminal does not stop it. Run it inside the LiLaWAM conda env.
#
# Usage:   bash scripts/train_bg.sh <config> [run_name] [key=value ...]
# Example: bash scripts/train_bg.sh configs/exp/baseline.yaml
#          bash scripts/train_bg.sh configs/exp/baseline.yaml "" experiment.seed=1
#
# Running the same command again after a stop/crash resumes from runs/<run_name>/latest.pt.
set -euo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SUB}"

CONFIG="${1:?usage: bash scripts/train_bg.sh <config> [run_name] [key=value ...]}"; shift
RUN_ARGS=()
if [[ $# -gt 0 && "$1" != *=* ]]; then
    [[ -n "$1" ]] && RUN_ARGS=(--run_name "$1")
    shift
fi

RUN_DIR="$(python scripts/step08_train.py --config "${CONFIG}" "${RUN_ARGS[@]}" --print_run_dir "$@" | tail -n 1)"
mkdir -p "${RUN_DIR}"

if [[ -f "${RUN_DIR}/train.pid" ]] && kill -0 "$(cat "${RUN_DIR}/train.pid")" 2>/dev/null; then
    echo "A training process ($(cat "${RUN_DIR}/train.pid")) is already running for ${RUN_DIR}; not starting a second one."
    exit 1
fi

[[ -f "${RUN_DIR}/latest.pt" ]] && echo "Found ${RUN_DIR}/latest.pt: this will RESUME that run."
nohup python -u scripts/step08_train.py --config "${CONFIG}" "${RUN_ARGS[@]}" "$@" \
    >> "${RUN_DIR}/train.log" 2>&1 &
echo $! > "${RUN_DIR}/train.pid"

cat <<EOF
Started in the background (PID $(cat "${RUN_DIR}/train.pid")).
  progress : tail -f ${RUN_DIR}/train.log
  pause    : touch ${RUN_DIR}/STOP          (saves, then exits after the current step)
  resume   : run this same command again
  GPU      : nvidia-smi
EOF
