#!/bin/bash
# Stage memory on all 10 tasks, warm-started from the RoboTwin checkpoint (configs/stage_memory/
# M_all_rt.yaml): 85,928 steps (round-1 recipe), then closed-loop evaluation with the time
# trigger. Unattended, in the background (LiLaWAM env, RMBENCH_DATA set).
#
#   bash scripts/step43_stage_all_rt.sh [test_num=50] [eval_seed=0]
#   (STAGE_CFG / STAGE_INIT override the config / initial weights)
#
# A separate copy of step42 so that the step-42 chain that is still running is not edited
# under it (bash reads a running script from disk). If any chain (runs/stage_all*.pid) is still
# running, this one waits for it first (one GPU).
# Order (log: runs/stage_all_<run>.log; rerun the same command to resume):
#   0. time triggers + data check   1. train (~19 h)   2. eval 10 tasks x test_num (~16 h)
#   3. analysis/stage_all_summary.py -> runs/stage_all_summary_<run>.md
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-50}"; SEED="${2:-0}"
cd "${SUB}"
CFG="${STAGE_CFG:-configs/stage_memory/M_all_rt.yaml}"
INIT="${STAGE_INIT:-/home/aki/lilawam/LiLa-WAM/checkpoints_vla/9task_visual/checkpoint_epoch_8.pt}"
RUN="$(python -c "from rmbench_lila.config import load_config; c=load_config('${CFG}', []); print(f'{c.experiment.name}-seed{c.experiment.seed}')")" \
    || { echo "cannot read ${CFG}"; exit 1; }
LOG="${SUB}/runs/stage_all_${RUN}.log"
PIDF="${SUB}/runs/stage_all_${RUN}.pid"
[[ -n "${RMBENCH_DATA:-}" ]] || { echo "set RMBENCH_DATA first"; exit 1; }
[[ -f "${INIT}" ]] || { echo "initial checkpoint missing: ${INIT}"; exit 1; }

if [[ "${_STAGE_ALL_CHILD:-}" != "1" ]]; then
    _STAGE_ALL_CHILD=1 STAGE_CFG="${CFG}" STAGE_INIT="${INIT}" nohup bash "$0" "$@" >> "${LOG}" 2>&1 &
    echo $! > "${PIDF}"
    echo "started in the background (PID $(cat "${PIDF}")); run ${RUN}; progress: tail -n 30 ${LOG}"
    exit 0
fi

say() { echo "----- $(date '+%F %T') $*"; }
last_step() { sed -E 's/step_0*([0-9]+)\.pt/\1/' "runs/$1/checkpoints/latest.txt" 2>/dev/null || echo 0; }

say "start: ${RUN} (${CFG}, init ${INIT})"
for f in runs/stage_all*.pid; do          # one GPU: wait for an earlier chain that is still running
    [[ -f "${f}" && "${f}" != "${PIDF}" ]] || continue
    p="$(cat "${f}")"
    while kill -0 "${p}" 2>/dev/null; do say "waiting for ${f} (PID ${p})"; sleep 600; done
done
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
python analysis/stage_all_summary.py runs "${RUN}" 2>&1 | tee "runs/stage_all_summary_${RUN}.md"
say "all done"
