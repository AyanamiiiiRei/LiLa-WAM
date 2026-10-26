#!/bin/bash
# Plan step 9: make the LiLa_WAM adapter visible to RMBench's XPolicyLab.
#
# Creates ONE symlink:  $RMBENCH_ROOT/XPolicyLab/policy/LiLa_WAM -> rmbench_exp/xpolicylab_adapter/LiLa_WAM
# Nothing is copied, so editing the adapter here takes effect immediately, and
# removing the symlink fully undoes the installation.
#
# Usage: RMBENCH_ROOT=/path/to/RMBench bash scripts/step09_install_adapter.sh [--uninstall]
set -euo pipefail

: "${RMBENCH_ROOT:?set RMBENCH_ROOT to your RMBench checkout}"
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SUB}/xpolicylab_adapter/LiLa_WAM"
DST="${RMBENCH_ROOT}/XPolicyLab/policy/LiLa_WAM"

if [[ "${1:-}" == "--uninstall" ]]; then
    [[ -L "${DST}" ]] && rm "${DST}" && echo "removed ${DST}" || echo "nothing to remove"
    exit 0
fi

[[ -d "${RMBENCH_ROOT}/XPolicyLab/policy" ]] || { echo "XPolicyLab submodule missing: git submodule update --init XPolicyLab"; exit 1; }
if [[ -e "${DST}" && ! -L "${DST}" ]]; then
    echo "${DST} exists and is not a symlink; refusing to overwrite"; exit 1
fi
ln -sfn "${SRC}" "${DST}"
chmod +x "${SRC}"/*.sh
echo "linked ${DST} -> ${SRC}"
cat <<EOF

Evaluate one task (from ${RMBENCH_ROOT}):
  TEST_NUM=100 bash XPolicyLab/policy/LiLa_WAM/eval.sh RMBench <task> <run, e.g. baseline/20261026-120000> \\
      aloha_agilex joint <seed> <policy_gpu> <env_gpu> <policy_conda_env> RMBench
Per-episode results: ${SUB}/runs/<run>/rmbench_eval/ckpt_<step>/episodes.jsonl
EOF
