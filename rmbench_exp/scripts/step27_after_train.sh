#!/bin/bash
# Stage-memory experiment: wait for a training run to finish, then run the post-training
# checks and the closed-loop evaluation automatically (in the background).
#
#   bash scripts/step27_after_train.sh <run> [test_num=50] [eval_seed=0] [compare_run=sm_A-seed0]
#   bash scripts/step27_after_train.sh sm_M0-seed0 50 0
#
# Start it in the LiLaWAM env while (or after) training runs. Steps, all logged to
# runs/<run>/after_train.log:
#   1. wait until the training process (runs/<run>/train.pid) has exited
#   2. stop here unless the last checkpoint is at training.total_steps (training finished;
#      a paused or crashed run is NOT evaluated)
#   3. held-out L_fm per stage           -> runs/<run>/heldout_by_stage.csv
#   4. slot-3 probe (step25), vs compare -> runs/<run>/slot_probe.txt
#   5. closed-loop eval with the oracle (step24), test_num episodes, eval_seed
#   6. failure classes (analysis/swap_blocks_progress.py) -> runs/<run>/failure_classes.txt
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: bash scripts/step27_after_train.sh <run> [test_num] [eval_seed] [compare_run]}"
N="${2:-50}"; SEED="${3:-0}"; CMP="${4:-sm_A-seed0}"
R="${SUB}/runs/${RUN}"
LOG="${R}/after_train.log"
[[ -d "${R}" ]] || { echo "no run ${R}"; exit 1; }

if [[ "${_AFTER_TRAIN_CHILD:-}" != "1" ]]; then
    _AFTER_TRAIN_CHILD=1 nohup bash "$0" "$@" >> "${LOG}" 2>&1 &
    echo $! > "${R}/after_train.pid"
    echo "Waiting for training, then checks + eval, in the background (PID $(cat "${R}/after_train.pid"))."
    echo "  progress: tail -f ${LOG}"
    echo "  cancel  : kill $(cat "${R}/after_train.pid")"
    exit 0
fi

cd "${SUB}"
echo "===== $(date '+%F %T') after_train: ${RUN} ====="
if [[ -f "${R}/train.pid" ]]; then
    TP="$(cat "${R}/train.pid")"
    while kill -0 "${TP}" 2>/dev/null; do sleep 60; done
fi
echo "$(date '+%F %T') training process has exited"

TOTAL="$(python -c "from omegaconf import OmegaConf; print(int(OmegaConf.load('${R}/config.yaml').training.total_steps))")"
LAST="$(sed -E 's/step_0*([0-9]+)\.pt/\1/' "${R}/checkpoints/latest.txt" 2>/dev/null || echo 0)"
if [[ "${LAST}" != "${TOTAL}" ]]; then
    echo "last checkpoint is step ${LAST}, training.total_steps is ${TOTAL}: training did not finish."
    echo "Not evaluating. Check ${R}/train.log, resume training, then start this script again."
    exit 1
fi

echo "----- $(date '+%F %T') held-out L_fm per stage (checkpoint ${LAST})"
python scripts/step23_stage_checks.py heldout --run "runs/${RUN}" || echo "held-out check FAILED (continuing)"

echo "----- $(date '+%F %T') slot-3 probe vs ${CMP}"
python scripts/step25_slot_probe.py --run "runs/${RUN}" --compare "runs/${CMP}" 2>&1 | tee "${R}/slot_probe.txt" \
    || echo "probe FAILED (continuing)"

echo "----- $(date '+%F %T') closed-loop eval: ${N} episodes, seed ${SEED}"
_EVAL_ALL_CHILD=1 bash scripts/step24_eval_with_oracle.sh "${RUN}" "${N}" "${SEED}"

echo "----- $(date '+%F %T') failure classes"
python analysis/swap_blocks_progress.py runs 2>&1 | tee "${R}/failure_classes.txt"
echo "===== $(date '+%F %T') after_train done ====="
