#!/bin/bash
# Sub-task design: quick end-to-end check BEFORE the unattended run (~20-30 min, foreground).
# Two tasks, 3 training steps per plan, 5 chooser steps, 1 closed-loop episode per plan.
#   bash scripts/step39_subtask_smoke.sh
# Pass = the last line says "SMOKE OK". Afterwards the smoke runs are deleted.
set -euo pipefail
SUB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SUB}"
INIT="${SUB}/runs/baseline-seed0/checkpoints/step_0085928.pt"
[[ -n "${RMBENCH_DATA:-}" ]] || { echo "set RMBENCH_DATA first"; exit 1; }
echo "== tests"; python -m pytest tests -q
echo "== CLS caches (2 tasks)"
for t in put_back_block swap_blocks; do python scripts/step21_cache_cls.py --task "$t"; done
echo "== library (2 tasks)"
python scripts/step36_subtask_library.py --tasks put_back_block swap_blocks --out artifacts/subtasks_smoke | tail -n 4
OV="rmbench.tasks=[put_back_block,swap_blocks] dataset.subtask.library_dir=artifacts/subtasks_smoke
    training.total_steps=3 training.save_every_steps=3 training.resume_every_steps=3 training.log_every_steps=1
    training.batch_size=16 training.warmup_steps=0"
rm -rf runs/smoke_P1 runs/smoke_P2
echo "== train P1 (joint) 3 steps"
python scripts/step08_train.py --config configs/subtask/P1.yaml ${OV} --run_name smoke_P1 --init_from "${INIT}" 2>&1 | tail -n 4
echo "== train P2 executor 3 steps"
python scripts/step08_train.py --config configs/subtask/P2.yaml ${OV} --run_name smoke_P2 --init_from "${INIT}" 2>&1 | tail -n 4
echo "== P2 chooser 5 steps + report"
python scripts/step38_train_chooser.py --run runs/smoke_P2 --steps 5 2>&1 | tail -n 6
echo "== P1 chooser report"
python scripts/step38_train_chooser.py --run runs/smoke_P1 --eval_only 2>&1 | tail -n 4
for r in smoke_P1 smoke_P2; do
    echo "== closed loop: $r, 1 episode of put_back_block"
    _EVAL_ALL_CHILD=1 bash scripts/step09_eval_all.sh "$r" 1 0 put_back_block > "runs/$r.eval.log" 2>&1 || { tail -n 30 "runs/$r.eval.log"; exit 1; }
    f="$(ls runs/$r/rmbench_eval/ckpt_*/episodes.jsonl)"
    python -c "import json,sys; e=[json.loads(l) for l in open('$f') if l.strip()][-1]; print('  success', e['success'], 'chunks', e['chunks'], 'sub-task switches', e['subtask_log']); sys.exit(0 if e['subtask_log'] is not None else 1)"
done
rm -rf runs/smoke_P1 runs/smoke_P2 runs/smoke_P1.eval.log runs/smoke_P2.eval.log artifacts/subtasks_smoke
echo "SMOKE OK"
