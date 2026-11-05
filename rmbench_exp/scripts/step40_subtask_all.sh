#!/bin/bash
# Sub-task design: everything, unattended, in the background (LiLaWAM env, RMBENCH_DATA set).
#
#   bash scripts/step40_subtask_all.sh [init_ckpt] [test_num=50] [eval_seed=0]
#   (init_ckpt default: runs/baseline-seed0/checkpoints/step_0085928.pt)
#
# Order (each step logs to runs/subtask_all.log; a failed step is logged and the next one runs):
#   0. CLS caches for all 10 tasks (step21) + sub-task library (step36)
#   1. train P1 (joint executor + chooser)           configs/subtask/P1.yaml
#   2. train P2 executor, then its chooser (step38)  configs/subtask/P2.yaml
#   3. closed-loop eval of P1, then P2: all 10 tasks x test_num, same seeds as round 1 (step09)
#   4. chooser report of P1 on held-out demos (step38 --eval_only)
#   5. analysis/subtask_summary.py -> runs/subtask_summary.md
# Rerun the same command to resume: finished trainings / evals are skipped.
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INIT="${1:-${SUB}/runs/baseline-seed0/checkpoints/step_0085928.pt}"
N="${2:-50}"; SEED="${3:-0}"
LOG="${SUB}/runs/subtask_all.log"
TASKS=(battery_try blocks_ranking_try cover_blocks observe_and_pickup place_block_mat press_button
       put_back_block rearrange_blocks swap_T swap_blocks)
cd "${SUB}"
[[ -n "${RMBENCH_DATA:-}" ]] || { echo "set RMBENCH_DATA first"; exit 1; }
[[ -f "${INIT}" ]] || { echo "no init checkpoint ${INIT}"; exit 1; }

if [[ "${_SUBTASK_CHILD:-}" != "1" ]]; then
    _SUBTASK_CHILD=1 nohup bash "$0" "$@" >> "${LOG}" 2>&1 &
    echo $! > "${SUB}/runs/subtask_all.pid"
    echo "started in the background (PID $(cat "${SUB}/runs/subtask_all.pid")); progress: tail -n 30 ${LOG}"
    exit 0
fi

say() { echo "----- $(date '+%F %T') $*"; }
total_steps() { python -c "from rmbench_lila.config import load_config; print(load_config('$1', []).training.total_steps)"; }
last_step() { sed -E 's/step_0*([0-9]+)\.pt/\1/' "runs/$1/checkpoints/latest.txt" 2>/dev/null || echo 0; }
train() {   # $1 config, $2 run name
    local cfg="$1" run="$2" tot
    tot="$(total_steps "${cfg}")"
    if [[ "$(last_step "${run}")" == "${tot}" ]]; then say "${run}: already trained"; return 0; fi
    say "train ${run} (${cfg})"
    if [[ -f "runs/${run}/latest.pt" ]]; then
        python -u scripts/step08_train.py --config "${cfg}" >> "runs/${run}.train.log" 2>&1
    else
        python -u scripts/step08_train.py --config "${cfg}" --init_from "${INIT}" >> "runs/${run}.train.log" 2>&1
    fi
    [[ "$(last_step "${run}")" == "${tot}" ]] || { say "${run}: training did not finish (see runs/${run}.train.log)"; return 1; }
}
evaluate() {  # $1 run
    say "eval $1: ${N} episodes x 10 tasks, seed ${SEED}"
    _EVAL_ALL_CHILD=1 bash scripts/step09_eval_all.sh "$1" "${N}" "${SEED}" || say "eval $1 FAILED"
}

say "start (init ${INIT})"
say "0. CLS caches + sub-task library"
for t in "${TASKS[@]}"; do
    python scripts/step21_cache_cls.py --task "${t}" || say "CLS ${t} FAILED"
done
[[ -f artifacts/subtasks/library.npz ]] || python scripts/step36_subtask_library.py || { say "library FAILED"; exit 1; }

train configs/subtask/P1.yaml sub_P1-seed0; P1_OK=$?
train configs/subtask/P2.yaml sub_P2-seed0; P2_OK=$?
if [[ ${P2_OK} == 0 && ! -f runs/sub_P2-seed0/chooser_report.md ]]; then
    say "train P2 chooser (frozen executor)"
    python -u scripts/step38_train_chooser.py --run runs/sub_P2-seed0 >> runs/sub_P2-seed0.chooser.log 2>&1 \
        || { say "P2 chooser FAILED"; P2_OK=1; }
fi
[[ ${P1_OK} == 0 ]] && evaluate sub_P1-seed0
[[ ${P2_OK} == 0 ]] && evaluate sub_P2-seed0
[[ ${P1_OK} == 0 && ! -f runs/sub_P1-seed0/chooser_report.md ]] && \
    { python -u scripts/step38_train_chooser.py --run runs/sub_P1-seed0 --eval_only >> runs/sub_P1-seed0.chooser.log 2>&1 || say "P1 chooser report FAILED"; }
say "summary"
python analysis/subtask_summary.py runs 2>&1 | tee runs/subtask_summary.md
say "all done"
