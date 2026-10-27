#!/bin/bash
# Plan step 9: evaluate one run on all RMBench tasks, in the background, resumable.
#
# Usage (from anywhere, LiLaWAM env active):
#   bash scripts/step09_eval_all.sh <run> [test_num] [eval_seed] [tasks...]
#   bash scripts/step09_eval_all.sh baseline-seed0 50
#   bash scripts/step09_eval_all.sh baseline-seed0 50 0 put_back_block swap_T     # only some tasks
#
# - Runs RMBench's eval.sh task by task with nohup; log: runs/<run>/rmbench_eval/eval_all.log
# - A task that already has >= test_num episodes in episodes.jsonl is skipped, so after an
#   interruption the same command continues with the unfinished tasks. (An interrupted task is
#   re-run from its first seed; analysis/results.py keeps one result per (task, seed).)
# - Stop after the current task:  touch runs/<run>/rmbench_eval/STOP_EVAL
# - The same eval_seed gives the same scenes for every model -> paired comparisons.
set -euo pipefail

SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMBENCH_ROOT="${RMBENCH_ROOT:-${HOME}/lilawam/RMBench}"
RUN="${1:?usage: bash scripts/step09_eval_all.sh <run> [test_num] [eval_seed] [tasks...]}"
TEST_NUM="${2:-50}"
EVAL_SEED="${3:-0}"
shift $(( $# < 3 ? $# : 3 ))
TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    # shortest first, so problems show up early
    TASKS=(observe_and_pickup put_back_block swap_T rearrange_blocks press_button swap_blocks
           battery_try cover_blocks place_block_mat blocks_ranking_try)
fi
POLICY_ENV="${POLICY_ENV:-LiLaWAM}"
EVAL_ENV="${EVAL_ENV:-RMBench}"
GPU="${GPU:-0}"

[[ -d "${SUB}/runs/${RUN}/checkpoints" ]] || { echo "no run ${SUB}/runs/${RUN}"; exit 1; }
[[ -L "${RMBENCH_ROOT}/XPolicyLab/policy/LiLa_WAM" ]] || { echo "adapter not linked: run scripts/step09_install_adapter.sh"; exit 1; }
CKPT_STEP="$(sed -E 's/step_0*([0-9]+)\.pt/\1/' "${SUB}/runs/${RUN}/checkpoints/latest.txt")"
OUT="${SUB}/runs/${RUN}/rmbench_eval"
mkdir -p "${OUT}"
LOG="${OUT}/eval_all.log"
PIDFILE="${OUT}/eval_all.pid"

if [[ "${_EVAL_ALL_CHILD:-}" != "1" ]]; then
    if [[ -f "${PIDFILE}" ]] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
        echo "evaluation already running (PID $(cat "${PIDFILE}")); tail -f ${LOG}"; exit 1
    fi
    rm -f "${OUT}/STOP_EVAL"
    _EVAL_ALL_CHILD=1 nohup bash "$0" "${RUN}" "${TEST_NUM}" "${EVAL_SEED}" "${TASKS[@]}" >> "${LOG}" 2>&1 &
    echo $! > "${PIDFILE}"
    cat <<EOF
Evaluation started in the background (PID $(cat "${PIDFILE}")): ${RUN}, ${TEST_NUM} episodes/task, seed ${EVAL_SEED}
  tasks    : ${TASKS[*]}
  progress : tail -f ${LOG}
  results  : python ${SUB}/analysis/results.py table ${SUB}/runs/${RUN}
  stop     : touch ${OUT}/STOP_EVAL   (after the current task; rerun the same command to continue)
EOF
    exit 0
fi

done_count() {   # unique seeds already evaluated for a task with this checkpoint
    local f="${OUT}/ckpt_${CKPT_STEP}/episodes.jsonl"
    [[ -f "$f" ]] || { echo 0; return; }
    python - "$f" "$1" <<'PY'
import json, sys
seeds = {json.loads(l)["seed"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["task"] == sys.argv[2]}
print(len(seeds))
PY
}

echo "===== $(date '+%F %T') eval_all: ${RUN} ckpt ${CKPT_STEP}, ${TEST_NUM} eps/task, seed ${EVAL_SEED} ====="
cd "${RMBENCH_ROOT}"
for task in "${TASKS[@]}"; do
    if [[ -f "${OUT}/STOP_EVAL" ]]; then
        echo "STOP_EVAL found: stopping before ${task}"; rm -f "${OUT}/STOP_EVAL"; break
    fi
    n="$(done_count "${task}")"
    if (( n >= TEST_NUM )); then
        echo "[skip] ${task}: ${n} episodes already"; continue
    fi
    echo "----- $(date '+%F %T') ${task} (have ${n}/${TEST_NUM}) -----"
    t0=$(date +%s)
    TEST_NUM="${TEST_NUM}" bash XPolicyLab/policy/LiLa_WAM/eval.sh RMBench "${task}" "${RUN}" \
        aloha_agilex joint "${EVAL_SEED}" "${GPU}" "${GPU}" "${POLICY_ENV}" "${EVAL_ENV}" \
        || echo "!!!!! ${task} exited with an error (continuing with the next task)"
    echo "----- ${task} done in $(( ($(date +%s) - t0) / 60 )) min, episodes now: $(done_count "${task}") -----"
done
echo "===== $(date '+%F %T') eval_all finished ====="
rm -f "${PIDFILE}"
