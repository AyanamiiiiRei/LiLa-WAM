#!/usr/bin/env python
"""Plan step 8: train LiLa-WAM on RMBench, resumable.

Run directory: runs/<run_name>, where run_name defaults to "<experiment.name>-seed<seed>".
Running the SAME command again resumes from runs/<run_name>/latest.pt (with the config
saved in the run directory). Pause safely with:  touch runs/<run_name>/STOP

Usage:
  python scripts/step08_train.py --config configs/exp/baseline.yaml
  python scripts/step08_train.py --config configs/exp/baseline.yaml experiment.seed=1      # another seed
  # in the background (survives closing the terminal; log in runs/<run_name>/train.log):
  bash scripts/train_bg.sh configs/exp/baseline.yaml
  # second LR stage, weights only, as a NEW run:
  python scripts/step08_train.py --config configs/exp/baseline.yaml --run_name baseline-stage2 \
      --init_from runs/baseline-seed0/checkpoints/step_0085928.pt training.learning_rate=5e-5 ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import _common
from omegaconf import OmegaConf

from rmbench_lila.config import validate
from rmbench_lila.trainer import Trainer, has_checkpoint

log = logging.getLogger("step08")


def run_dir_for(cfg, run_name: str | None) -> Path:
    name = run_name or f"{cfg.experiment.name}-seed{cfg.experiment.seed}"
    return Path(cfg.experiment.output_root) / name


def main():
    p = _common.parser(__doc__)
    p.add_argument("--run_name", default=None, help="default <experiment.name>-seed<seed>")
    p.add_argument("--init_from", default=None, help="start a NEW run from these weights")
    p.add_argument("--device", default="cuda")
    p.add_argument("--print_run_dir", action="store_true", help="print the run directory and exit")
    args = p.parse_args()
    cfg = _common.load(args)
    run_dir = run_dir_for(cfg, args.run_name)
    if args.print_run_dir:
        print(run_dir)
        return

    if has_checkpoint(run_dir):
        saved = OmegaConf.load(run_dir / "config.yaml")
        validate(saved)
        if OmegaConf.to_yaml(saved) != OmegaConf.to_yaml(cfg):
            log.warning("run directory exists: resuming with ITS saved config; the given --config/overrides "
                        "differ and are ignored (use another --run_name for a new run)")
        cfg = saved
    device, dtype = _common.pick_device_dtype(args.device)
    status = Trainer(cfg, run_dir, device, dtype, init_from=args.init_from).train()
    print(f"\n{status}. Run directory: {run_dir}")
    sys.exit(0)


if __name__ == "__main__":
    main()
