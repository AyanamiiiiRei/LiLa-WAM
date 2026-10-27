#!/bin/bash
# Plan step 9: evaluate one trained run on all RMBench tasks, one task after another.
#
# Usage (from anywhere; runs unattended, survives closing the terminal):
#   nohup bash scripts/step09_eval_all.sh baseline-seed0 100 > runs/baseline-seed0/rmbench_eval/eval_all.log 2>&1 &
#
#   arg 1  run directory name under runs/            (e.g. baseline-seed0)
#   arg 2  episodes per task (TEST_NUM, default 100)
# Environment overrides: RMBENCH_ROOT (~/lilawam/RMBench), TASKS ("a b c"), EVAL_SEED (0),
#   POLICY_ENV (LiLaWAM), EVAL_ENV (RMBench), POLICY_GPU (0), ENV_GPU (0)
#
# Shortest tasks run first. A task whose episodes.jsonl already holds TEST_NUM distinct seeds is
# skipped, so after an interruption the same command continues with the unfinished tasks
# (an interrupted task restarts from its first episode; analysis keeps one result per seed).
# Stop after the current task:  touch runs/<run>/rmbench_eval/STOP_EVAL
set -uo pipefail

SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: step09_eval_all.sh <run> [TEST_NUM]}"
TEST_NUM="${2:-100}"
RMBENCH_ROOT="${RMBENCH_ROOT:-${HOME}/lilawam/RMBench}"
EVAL_SEED="${EVAL_SEED:-0}"
POLICY_ENV="${POLICY_ENV:-LiLaWAM}"
EVAL_ENV="${EVAL_ENV:-RMBench}"
POLICY_GPU="${POLICY_GPU:-0}"
ENV_GPU="${ENV_GPU:-0}"
# ordered by step limit (env_cfg/task_config/_eval_step_limit.yml), shortest first
TASKS="${TASKS:-observe_and_pickup put_back_block swap_T rearrange_blocks battery_try swap_blocks cover_blocks place_block_mat press_button blocks_ranking_try}"

RUN_DIR="${SUB}/runs/${RUN}"
[[ -f "${RUN_DIR}/checkpoints/latest.txt" ]] || { echo "no checkpoints in ${RUN_DIR}"; exit 1; }
[[ -L "${RMBENCH_ROOT}/XPolicyLab/policy/LiLa_WAM" ]] || { echo "adapter not linked: run scripts/step09_install_adapter.sh"; exit 1; }
STEP="$(sed -E 's/[^0-9]//g; s/^0+//' "${RUN_DIR}/checkpoints/latest.txt")"
EVAL_DIR="${RUN_DIR}/rmbench_eval"
JSONL="${EVAL_DIR}/ckpt_${STEP}/episodes.jsonl"
mkdir -p "${EVAL_DIR}/logs"
rm -f "${EVAL_DIR}/STOP_EVAL"

done_seeds() {   # distinct seeds already evaluated for a task
    [[ -f "${JSONL}" ]] || { echo 0; return; }
    python3 - "$JSONL" "$1" <<'PY'
import json, sys
seeds = {json.loads(l)["seed"] for l in open(sys.argv[1]) if l.strip() and json.loads(l)["task"] == sys.argv[2]}
print(len(seeds))
PY
}

echo "[eval_all] run=${RUN} ckpt_step=${STEP} test_num=${TEST_NUM} eval_seed=${EVAL_SEED} $(date '+%F %T')"
cd "${RMBENCH_ROOT}"
for task in ${TASKS}; do
    if [[ -f "${EVAL_DIR}/STOP_EVAL" ]]; then
        echo "[eval_all] STOP_EVAL found, stopping before ${task}"; rm -f "${EVAL_DIR}/STOP_EVAL"; exit 0
    fi
    n="$(done_seeds "${task}")"
    if (( n >= TEST_NUM )); then
        echo "[eval_all] ${task}: ${n} episodes already done, skipping"; continue
    fi
    echo "[eval_all] ${task}: start ($(date '+%F %T')), log ${EVAL_DIR}/logs/${task}.log"
    TEST_NUM="${TEST_NUM}" bash XPolicyLab/policy/LiLa_WAM/eval.sh RMBench "${task}" "${RUN}" \
        aloha_agilex joint "${EVAL_SEED}" "${POLICY_GPU}" "${ENV_GPU}" "${POLICY_ENV}" "${EVAL_ENV}" \
        > "${EVAL_DIR}/logs/${task}.log" 2>&1
    status=$?
    echo "[eval_all] ${task}: exit ${status}, $(done_seeds "${task}") episodes recorded ($(date '+%F %T'))"
    grep -a "Final success rate" "${EVAL_DIR}/logs/${task}.log" | tail -1
done
echo "[eval_all] finished $(date '+%F %T')"
echo "results: cd ${SUB} && python analysis/results.py table runs/${RUN}"
