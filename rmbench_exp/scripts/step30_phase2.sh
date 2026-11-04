#!/bin/bash
# Stage-memory phase 2, all steps in the background (LiLaWAM env, RMBENCH_DATA set):
#
#   bash scripts/step30_phase2.sh [run=sm_M-seed0] [test_num=50] [eval_seed=0] [triggers="proprio head time"]
#
#   1. step28: switch-head features on the 50 demos (GPU, the M checkpoint)
#   2. step29: train the switch head, offline report, time_trigger.json
#   3. closed-loop eval of each trigger with the SAME checkpoint and seeds as M (step24)
#   4. analysis/trigger_vs_oracle.py + swap_blocks_progress.py -> runs/<run>/phase2_report.txt
# Log: runs/<run>/phase2.log. A finished eval is skipped when rerun (step09 counts episodes).
set -uo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-sm_M-seed0}"; N="${2:-50}"; SEED="${3:-0}"; TRIGS="${4:-proprio head time}"
R="${SUB}/runs/${RUN}"; LOG="${R}/phase2.log"; TASK=swap_blocks
[[ -d "${R}" ]] || { echo "no run ${R}"; exit 1; }
[[ -n "${RMBENCH_DATA:-}" ]] || { echo "set RMBENCH_DATA first"; exit 1; }

if [[ "${_PHASE2_CHILD:-}" != "1" ]]; then
    _PHASE2_CHILD=1 nohup bash "$0" "${RUN}" "${N}" "${SEED}" "${TRIGS}" >> "${LOG}" 2>&1 &
    echo $! > "${R}/phase2.pid"
    echo "phase 2 started in the background (PID $(cat "${R}/phase2.pid")); progress: tail -f ${LOG}"
    exit 0
fi

cd "${SUB}"
ART="artifacts/stage_memory/${TASK}"
echo "===== $(date '+%F %T') phase 2: ${RUN}, triggers: ${TRIGS} ====="
if [[ ! -f "${ART}/switch_head.pt" ]]; then
    echo "----- $(date '+%F %T') step28 features"
    python scripts/step28_switch_features.py --run "runs/${RUN}" || { echo "step28 FAILED"; exit 1; }
    echo "----- $(date '+%F %T') step29 switch head"
    python scripts/step29_switch_head.py || { echo "step29 FAILED"; exit 1; }
fi
for trig in ${TRIGS}; do
    echo "----- $(date '+%F %T') closed-loop eval, trigger=${trig}"
    _EVAL_ALL_CHILD=1 bash scripts/step24_eval_with_oracle.sh "${RUN}" "${N}" "${SEED}" normal "${TASK}" "${trig}" \
        || echo "eval ${trig} FAILED (continuing)"
done
echo "----- $(date '+%F %T') analysis"
{ python analysis/trigger_vs_oracle.py runs; echo; python analysis/swap_blocks_progress.py runs; } 2>&1 | tee "${R}/phase2_report.txt"
echo "===== $(date '+%F %T') phase 2 done ====="
