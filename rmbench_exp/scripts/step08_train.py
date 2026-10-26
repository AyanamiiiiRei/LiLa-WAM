#!/usr/bin/env python
"""Plan step 8: train LiLa-WAM on RMBench.

Usage:
  python scripts/step08_train.py --config configs/exp/baseline.yaml
  python scripts/step08_train.py --config configs/exp/no_foresight.yaml experiment.seed=1
  # second LR stage (as in the paper: 2e-4 then 5e-5), weights only:
  python scripts/step08_train.py --config configs/exp/baseline.yaml --init_from runs/.../step_0030000.pt \
      training.learning_rate=5e-5 training.lr_min=5e-6 training.total_steps=10000 experiment.name=baseline_s2
  # continue an interrupted run (same config!):
  python scripts/step08_train.py --config runs/baseline/<ts>/config.yaml --resume runs/baseline/<ts>/checkpoints/step_0010000.pt

Run directory: runs/<experiment.name>/<timestamp>/
"""

from __future__ import annotations

from pathlib import Path

import _common

from rmbench_lila.trainer import Trainer
from rmbench_lila.utils.repro import timestamp


def main():
    p = _common.parser(__doc__)
    p.add_argument("--init_from", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--run_dir", default=None, help="default runs/<name>/<timestamp>")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = _common.load(args)
    device, dtype = _common.pick_device_dtype(args.device)
    run_dir = Path(args.run_dir) if args.run_dir else (
        Path(cfg.experiment.output_root) / cfg.experiment.name / timestamp())
    if args.resume and not args.run_dir:
        run_dir = Path(args.resume).resolve().parents[1]      # keep writing into the same run
    Trainer(cfg, run_dir, device, dtype, init_from=args.init_from, resume=args.resume).train()
    print(f"\nDone. Run directory: {run_dir}")


if __name__ == "__main__":
    main()
