#!/usr/bin/env python
"""Stage-memory phase 3, option A: train the learned-switch model (rmbench_lila/learned_switch.py).

  python scripts/step32_train_switch.py --config configs/stage_memory/S.yaml \\
      --init_from /home/aki/lilawam/LiLa-WAM/checkpoints_vla/9task_visual/checkpoint_epoch_8.pt
  # background: nohup python -u scripts/step32_train_switch.py ... >> runs/sm_S-seed0/train.log 2>&1 &

Run directory: runs/<experiment.name>-seed<seed>. Pause: touch runs/<run>/STOP (saves latest.pt);
the same command without --init_from resumes. Per-step log: runs/<run>/switch_train.csv
(losses, sum of p, number of switches, switch chunks of the first demo of the step).
"""

from __future__ import annotations

from pathlib import Path

import _common


def main():
    p = _common.parser(__doc__)
    p.add_argument("--init_from", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--print_run_dir", action="store_true")
    args = p.parse_args()
    cfg = _common.load(args)
    from rmbench_lila.learned_switch import train
    run = Path(cfg.experiment.output_root) / (args.run_name or f"{cfg.experiment.name}-seed{cfg.experiment.seed}")
    if args.print_run_dir:
        print(run)
        return
    device, dtype = _common.pick_device_dtype("cuda")
    print(f"{train(cfg, run, device, dtype, args.init_from)}. Run directory: {run}")


if __name__ == "__main__":
    main()
