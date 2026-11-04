#!/bin/bash
# Stage-memory phase 3 (option A), everything in the background (LiLaWAM env, RMBENCH_DATA set):
#
#   bash scripts/step34_phase3.sh <init_checkpoint> [test_num=50] [eval_seed=0]
#
#   1. train the learned-switch model (configs/stage_memory/S.yaml, step32); resumes if stopped
#   2. offline: where the switches fire on the 50 demos (step33)
#   3. closed-loop eval, 50 episodes, same seeds as M (step24; the oracle only logs)
#   4. analysis/trigger_vs_oracle.py + swap_blocks_progress.py -> runs/sm_S-seed0/phase3_report.txt
# Log: runs/sm_S-seed0/phase3.log (training itself: runs/sm_S-seed0/train.log)
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT="${1:?usage: bash scripts/step34_phase3.sh <init_checkpoint> [test_num] [eval_seed]}"
N="${2:-50}"; SEED="${3:-0}"
CFG=configs/stage_memory/S.yaml
cd "${SUB}"
[[ -n "${RMBENCH_DATA:-}" ]] || { echo "set RMBENCH_DATA first"; exit 1; }
RUN_DIR="$(python scripts/step32_train_switch.py --config "${CFG}" --print_run_dir | tail -n 1)"
RUN="$(basename "${RUN_DIR}")"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/phase3.log"

if [[ "${_PHASE3_CHILD:-}" != "1" ]]; then
    _PHASE3_CHILD=1 nohup bash "$0" "$@" >> "${LOG}" 2>&1 &
    echo $! > "${RUN_DIR}/phase3.pid"
    echo "phase 3 started in the background (PID $(cat "${RUN_DIR}/phase3.pid"))"
    echo "  overall : tail -n 20 ${LOG}"
    echo "  training: tail -n 5 ${RUN_DIR}/train.log"
    echo "  pause   : touch ${RUN_DIR}/STOP   (then run this command again to resume)"
    exit 0
fi

echo "===== $(date '+%F %T') phase 3: ${RUN} ====="
TOTAL="$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('${CFG}').model.learned_switch.total_steps)")"
last_step() { sed -E 's/step_0*([0-9]+)\.pt/\1/' "${RUN_DIR}/checkpoints/latest.txt" 2>/dev/null || echo 0; }
if [[ "$(last_step)" != "${TOTAL}" ]]; then
    echo "----- $(date '+%F %T') training (log: ${RUN_DIR}/train.log)"
    if [[ -f "${RUN_DIR}/latest.pt" ]]; then
        python -u scripts/step32_train_switch.py --config "${CFG}" >> "${RUN_DIR}/train.log" 2>&1
    else
        python -u scripts/step32_train_switch.py --config "${CFG}" --init_from "${INIT}" >> "${RUN_DIR}/train.log" 2>&1
    fi
    if [[ "$(last_step)" != "${TOTAL}" ]]; then
        echo "training stopped at step $(last_step) of ${TOTAL}; see train.log. Run this script again to resume."
        exit 1
    fi
fi
echo "----- $(date '+%F %T') offline switches on the demos"
python scripts/step33_switch_offline.py --run "runs/${RUN}" || echo "step33 FAILED (continuing)"
echo "----- $(date '+%F %T') closed-loop eval: ${N} episodes, seed ${SEED}"
_EVAL_ALL_CHILD=1 bash scripts/step24_eval_with_oracle.sh "${RUN}" "${N}" "${SEED}" || echo "eval FAILED"
echo "----- $(date '+%F %T') analysis"
{ cat "${RUN_DIR}/switch_offline.md" 2>/dev/null; echo; python analysis/trigger_vs_oracle.py runs; echo; python analysis/swap_blocks_progress.py runs; } 2>&1 \
    | tee "${RUN_DIR}/phase3_report.txt"
echo "===== $(date '+%F %T') phase 3 done ====="
