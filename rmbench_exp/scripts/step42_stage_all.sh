#!/bin/bash
# Stage memory on all 10 tasks: warm start from baseline-seed0, 8,000 steps, then closed-loop
# evaluation with the time trigger. Unattended, in the background (LiLaWAM env, RMBENCH_DATA set).
#
#   bash scripts/step42_stage_all.sh [test_num=50] [eval_seed=0]
#
# Order (log: runs/stage_all.log; rerun the same command to resume, finished steps are skipped):
#   0. time triggers + data check (step42_time_triggers.py); stops here if the check fails
#   1. train sm_Mall-seed0 (configs/stage_memory/M_all.yaml), init from baseline-seed0 step 85928
#   2. closed-loop eval, 10 tasks x test_num, same seeds as the baseline, trigger = time
#      -> runs/sm_Mall-seed0/rmbench_eval/ckpt_8000_M-time
#   3. analysis/stage_all_summary.py -> runs/stage_all_summary.md
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-50}"; SEED="${2:-0}"
CFG=configs/stage_memory/M_all.yaml
RUN=sm_Mall-seed0
INIT=runs/baseline-seed0/checkpoints/step_0085928.pt
LOG="${SUB}/runs/stage_all.log"
cd "${SUB}"
[[ -n "${RMBENCH_DATA:-}" ]] || { echo "set RMBENCH_DATA first"; exit 1; }
[[ -f "${INIT}" ]] || { echo "baseline checkpoint missing: ${INIT}"; exit 1; }

if [[ "${_STAGE_ALL_CHILD:-}" != "1" ]]; then
    _STAGE_ALL_CHILD=1 nohup bash "$0" "$@" >> "${LOG}" 2>&1 &
    echo $! > "${SUB}/runs/stage_all.pid"
    echo "started in the background (PID $(cat "${SUB}/runs/stage_all.pid")); progress: tail -n 30 ${LOG}"
    exit 0
fi

say() { echo "----- $(date '+%F %T') $*"; }
last_step() { sed -E 's/step_0*([0-9]+)\.pt/\1/' "runs/$1/checkpoints/latest.txt" 2>/dev/null || echo 0; }

say "start: ${RUN}"
say "0. time triggers + data check"
python -u scripts/step42_time_triggers.py --config "${CFG}" || { say "data check FAILED, stopping"; exit 1; }

TOT="$(python -c "from rmbench_lila.config import load_config; print(load_config('${CFG}', []).training.total_steps)")"
if [[ "$(last_step "${RUN}")" == "${TOT}" ]]; then
    say "1. ${RUN}: already trained"
else
    say "1. train ${RUN} (${TOT} steps, init ${INIT})"
    if [[ -f "runs/${RUN}/latest.pt" ]]; then      # resume: --init_from only starts NEW runs
        python -u scripts/step08_train.py --config "${CFG}" >> "runs/${RUN}.train.log" 2>&1
    else
        python -u scripts/step08_train.py --config "${CFG}" --init_from "${INIT}" >> "runs/${RUN}.train.log" 2>&1
    fi
    [[ "$(last_step "${RUN}")" == "${TOT}" ]] || { say "training did not finish (see runs/${RUN}.train.log)"; exit 1; }
fi

say "2. eval ${RUN}: ${N} episodes x 10 tasks, seed ${SEED}, trigger time"
LILA_STAGE_TRIGGER=time LILA_EVAL_TAG=M-time LILA_TIME_TRIGGER_FILE=time_trigger_10task.json \
    _EVAL_ALL_CHILD=1 bash scripts/step09_eval_all.sh "${RUN}" "${N}" "${SEED}" || say "eval FAILED"

say "3. summary"
python analysis/stage_all_summary.py runs 2>&1 | tee runs/stage_all_summary.md
say "all done"
