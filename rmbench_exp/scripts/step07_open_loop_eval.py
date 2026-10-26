#!/usr/bin/env python
"""Plan step 7.5: open-loop evaluation on held-out demonstrations (no simulator needed).

For every held-out episode, the policy is fed the recorded states/images step by step
(exactly as the RMBench adapter would) and, every --stride steps, predicts a chunk that
is compared with the recorded actions.

Reported per task: mean absolute error (rad) over the executed part (first
action_execution_horizon steps) and over the full chunk, separately for arm joints
and grippers.

Usage:
  python scripts/step07_open_loop_eval.py --run runs/baseline/<ts> [--checkpoint latest] [--stride 16]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import _common
import numpy as np
import torch

from rmbench_lila.episodes import EpisodeReader, clip_indices, select_episodes
from rmbench_lila.policy import LiLaWAMPolicy
from rmbench_lila.schema import STATE_GROUP, get_action_layout, get_state_layout


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", required=True)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device, dtype = _common.pick_device_dtype(args.device)
    pol = LiLaWAMPolicy.from_run(args.run, args.checkpoint, device, dtype)
    cfg = pol.cfg
    a_lay, s_lay = get_action_layout(cfg.rmbench.action_layout), get_state_layout(cfg.rmbench.state_layout)
    cam = list(cfg.dataset.camera_names)[0]
    offs = list(cfg.dataset.indices_config.action_indices)
    H = int(cfg.common.action_execution_horizon)
    arm = np.array(a_lay.angle_dims)
    grip = np.array(a_lay.gripper_dims)
    g = torch.Generator(device=device).manual_seed(args.seed)

    refs = select_episodes(cfg, "holdout")[: args.max_episodes]
    if not refs:
        raise SystemExit("No held-out episodes (rmbench.holdout_per_task = 0 in this run's config).")
    rows = defaultdict(list)
    curves = {}
    for ref in refs:
        pol.set_task(ref.task)
        pol.reset()
        with EpisodeReader(ref.path) as rd:
            states = rd.read_vector(s_lay, STATE_GROUP)
            actions = rd.read_action(a_lay, cfg.rmbench.action_alignment)
            n = states.shape[0]
            for t in range(n):
                anchor = t % args.stride == 0
                pol.observe(states[t], rd.read_frames(cam, [t])[0] if anchor else None)
                if not anchor:
                    continue
                pred = pol.predict_chunk(g)
                idx, valid = clip_indices(t, offs, n)
                err = np.abs(pred - actions[idx])
                vh = valid.copy()
                vh[H:] = False
                rows[ref.task].append({
                    "arm_exec": float(err[vh][:, arm].mean()), "grip_exec": float(err[vh][:, grip].mean()),
                    "arm_full": float(err[valid][:, arm].mean()), "grip_full": float(err[valid][:, grip].mean()),
                })
                if ref.task not in curves:
                    curves[ref.task] = None
                if curves[ref.task] is None or curves[ref.task]["episode_id"] == ref.episode_id:
                    c = curves[ref.task] or {"episode_id": ref.episode_id, "gt": actions.tolist(), "pred": []}
                    c["pred"].append([t, pred[:H].tolist()])
                    curves[ref.task] = c
        print(f"{ref.task} ep{ref.episode_id}: {len(rows[ref.task])} chunks so far")

    summary = {t: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} | {"chunks": len(rs)}
               for t, rs in sorted(rows.items())}
    out = Path(args.run) / "open_loop" / f"ckpt_{pol.checkpoint_step}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "curves.json").write_text(json.dumps(curves))
    print(f"\n{'task':24s} {'arm exec':>9s} {'grip exec':>9s} {'arm full':>9s} {'grip full':>9s}  (MAE, rad / gripper units)")
    for t, s in summary.items():
        print(f"{t:24s} {s['arm_exec']:9.4f} {s['grip_exec']:9.4f} {s['arm_full']:9.4f} {s['grip_full']:9.4f}")
    _plot(curves, out)
    print(f"\nSaved to {out}")


def _plot(curves, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task, c in curves.items():
        gt = np.array(c["gt"])
        fig, axes = plt.subplots(gt.shape[1], 1, figsize=(12, 1.4 * gt.shape[1]), sharex=True)
        for d, ax in enumerate(axes):
            ax.plot(gt[:, d], "k-", lw=1)
            for t, pred in c["pred"]:
                pr = np.array(pred)
                ax.plot(range(t, t + len(pr)), pr[:, d], "r-", lw=1, alpha=0.8)
            ax.set_ylabel(str(d), fontsize=7)
        axes[0].set_title(f"{task} ep{c['episode_id']}: black = recorded, red = predicted (executed part)")
        fig.tight_layout()
        fig.savefig(out / f"{task}.png", dpi=90)
        plt.close(fig)


if __name__ == "__main__":
    main()
