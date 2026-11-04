#!/usr/bin/env python
"""Stage-memory phase 2: offline inputs for the learned switch head (rmbench_lila/triggers.py).

  python scripts/step28_switch_features.py --run runs/sm_M-seed0

For every demo (45 train + 5 held-out) and every chunk start t = 0, 16, 32, ... (the moments
the policy decides at), with the M model and the TRUE frame/state at t:
  s(t)  = stages ended at or before t (boundaries_gt.json grid ends; same as the oracle,
          verified by step26)
  for every k = 0 .. min(s(t), 2) (memory with the first k slots written correctly, i.e. the
  trigger is on time (k = s) or late (k < s)):
      cls    last-layer DINOv3 CLS of the frame
      delta  foresight local change: mean predicted future patch feature (first flow step,
             with this memory) - mean current patch feature of the same layer
      label  1 if s(t) > k ("the current stage has ended -> write now"), else 0
States with k > s(t) (a premature write) are not generated: their memory content would
depend on when the trigger fired. Output: artifacts/stage_memory/<task>/switch_features/<ep>.npz
"""

from __future__ import annotations

from pathlib import Path

import _common
import numpy as np
import torch
from omegaconf import OmegaConf

from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.policy import LiLaWAMPolicy
from rmbench_lila.schema import STATE_GROUP, get_state_layout
from rmbench_lila.stages import GRID, load_boundaries


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", required=True)
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    bounds = load_boundaries(cfg.dataset.stage_memory.boundaries)
    cls_dir = Path(cfg.dataset.stage_memory.cls_dir)
    cam = list(cfg.dataset.camera_names)[0]
    s_layout = get_state_layout(cfg.rmbench.state_layout)
    held = {r.path for r in select_episodes(cfg, "holdout")}
    refs = select_episodes(cfg, "all")
    out = Path(args.out or f"artifacts/stage_memory/{args.task}/switch_features")
    out.mkdir(parents=True, exist_ok=True)

    pol = LiLaWAMPolicy.from_run(run, "latest", args.device)
    pol.set_task(args.task)
    m = pol.memory
    n_slots = m.num_slots
    for ref in refs:
        name = Path(ref.path).name
        ends = [s[1] for s in bounds[name][:n_slots]]
        starts = [0] + ends[:-1]
        cls = np.load(cls_dir / (Path(ref.path).stem + ".npy")).astype(np.float32)
        with EpisodeReader(ref.path) as r:
            n = r.length(s_layout, STATE_GROUP)
            states = r.read_vector(s_layout, STATE_GROUP)
            ts = list(range(0, n, GRID))
            frames = r.read_frames(cam, ts)
        rows = {"t": [], "k": [], "s": [], "label": [], "cls": [], "delta": []}
        for t, rgb in zip(ts, frames):
            s = sum(t >= e for e in ends)
            for k in range(0, min(s, n_slots - 1) + 1):
                pol.reset()
                pol.observe(states[t], rgb)
                m.cls[:] = 0
                m.written[:] = False
                for j in range(k):
                    m.cls[j, 0], m.cls[j, 1] = cls[starts[j]], cls[ends[j]]
                    m.written[j] = True
                g = torch.Generator(device=pol.device).manual_seed(1000 + t)
                c, d = pol.stage_features(g)
                for key, v in (("t", t), ("k", k), ("s", s), ("label", int(s > k)), ("cls", c), ("delta", d)):
                    rows[key].append(v)
        np.savez_compressed(out / (Path(ref.path).stem + ".npz"), split="heldout" if ref.path in held else "train",
                            ends=np.array(ends), length=n,
                            **{k: np.asarray(v, np.float32 if k in ("cls", "delta") else np.int64)
                               for k, v in rows.items()})
        print(f"{name}: {len(rows['t'])} samples, positives {sum(rows['label'])}, ends {ends}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
