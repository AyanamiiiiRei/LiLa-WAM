#!/bin/bash
# Stage-memory experiment: closed-loop evaluation WITH the simulator stage oracle.
#
#   bash scripts/step24_eval_with_oracle.sh <run> [test_num=100] [eval_seed=0] [memory=normal|empty] [task=swap_blocks]
#   bash scripts/step24_eval_with_oracle.sh sm_M-seed0 100 0            # group M
#   bash scripts/step24_eval_with_oracle.sh sm_M-seed0 100 0 empty      # group M-empty (same checkpoint)
#   bash scripts/step24_eval_with_oracle.sh sm_A-seed0 100 0            # groups A and F (oracle unused, same seeds)
#
# Same as step09_eval_all.sh (background, resumable, same seeds -> paired comparison), plus:
#   LILA_STAGE_ORACLE=1 + eval_hooks/ on PYTHONPATH: the RMBench client adds obs["oracle"]
#   (eval_hooks/sitecustomize.py, rmbench_lila/oracle.py); nothing in RMBench is edited.
#   memory=empty: the policy never writes memory; results go to rmbench_eval/ckpt_<step>_M-empty.
set -euo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?usage: bash scripts/step24_eval_with_oracle.sh <run> [test_num] [eval_seed] [normal|empty] [task]}"
N="${2:-100}"; SEED="${3:-0}"; MEM="${4:-normal}"; TASK="${5:-swap_blocks}"
export LILA_STAGE_ORACLE=1
export PYTHONPATH="${SUB}/eval_hooks${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${MEM}" == "empty" ]]; then
    export LILA_MEMORY_OVERRIDE=empty LILA_EVAL_TAG=M-empty
else
    unset LILA_MEMORY_OVERRIDE LILA_EVAL_TAG || true
fi
exec bash "${SUB}/scripts/step09_eval_all.sh" "${RUN}" "${N}" "${SEED}" "${TASK}"
