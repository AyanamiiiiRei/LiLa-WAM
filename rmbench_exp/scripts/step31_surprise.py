#!/usr/bin/env python
"""Stage-memory phase 3 (diagnostic C): the model's own foresight "surprise" on the demos.

  python scripts/step31_surprise.py --run runs/sm_A-seed0

Question: is stage structure already present in what the trained model predicts? For every
demo (45 train + 5 held-out) and every chunk start t = 0, 16, 32, ... of a run WITHOUT memory
(default: group A, the original LiLa-WAM trained on swap_blocks):
  err    foresight surprise = mean over patches of (1 - cos(predicted, actual)) for the frame
         t + future_frame_offset (32), i.e. the model's own foresight loss at t
         (first flow step t=0, fixed noise, true frame and state at t)
  delta  foresight local change = mean predicted future patch feature - mean current patch
         feature (same DINOv3 layer); used by the "foresight change" baseline
Nothing is trained. Output: artifacts/stage_memory/<task>/surprise/<run name>/<episode>.npz;
analysed by analysis/stage_discovery.py.
"""

from __future__ import annotations

from pathlib import Path

import _common
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from rmbench_lila import lila_api
from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.image import to_model_input
from rmbench_lila.policy import LiLaWAMPolicy
from rmbench_lila.schema import STATE_GROUP, get_state_layout
from rmbench_lila.stages import GRID


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", default="runs/sm_A-seed0")
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    off = cfg.dataset.get("future_frame_offset") or cfg.common.action_chunk_size
    cam = list(cfg.dataset.camera_names)[0]
    s_layout = get_state_layout(cfg.rmbench.state_layout)
    held = {r.path for r in select_episodes(cfg, "holdout")}
    refs = select_episodes(cfg, "all")
    out = Path(f"artifacts/stage_memory/{args.task}/surprise/{run.name}")
    out.mkdir(parents=True, exist_ok=True)

    pol = LiLaWAMPolicy.from_run(run, "latest", args.device)
    if pol.memory is not None:
        raise SystemExit("use a run without stage memory (e.g. runs/sm_A-seed0): the surprise must not "
                         "depend on stage boundaries")
    pol.set_task(args.task)
    size = pol.image_size
    for ref in refs:
        with EpisodeReader(ref.path) as r:
            n = r.length(s_layout, STATE_GROUP)
            states = r.read_vector(s_layout, STATE_GROUP)
            ts = list(range(0, n, GRID))
            fut = [min(t + int(off), n - 1) for t in ts]
            frames = r.read_frames(cam, ts)
            fframes = r.read_frames(cam, fut)
        errs, deltas = [], []
        with torch.no_grad():
            for t, rgb, frgb in zip(ts, frames, fframes):
                pol.reset()
                pol.observe(states[t], rgb)
                g = torch.Generator(device=pol.device).manual_seed(1000 + t)
                px, qpos, feats, x, tsteps = pol._inputs(g)
                o = lila_api.forward_full(pol.action_model, tsteps[0][None], x, qpos, feats, pol.task_cond)
                pred = pol.action_model.future_feat_decoder(o["cond_tokens"]).float()        # (1, P, D)
                fpx = torch.from_numpy(to_model_input(frgb, size)).to(pol.device, pol.dtype)[None]
                target = pol.model.get_future_target_features(fpx).float()
                cur = pol.model.get_future_target_features(px).float()
                cos = (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(-1)    # (1, P)
                errs.append(float((1 - cos).mean()))
                deltas.append((pred.mean(1) - cur.mean(1))[0].cpu().numpy())
        np.savez_compressed(out / (Path(ref.path).stem + ".npz"), t=np.array(ts), err=np.array(errs, np.float32),
                            delta=np.array(deltas, np.float32), length=n,
                            split="heldout" if ref.path in held else "train")
        print(f"{Path(ref.path).name}: {len(ts)} chunks, surprise mean {np.mean(errs):.4f} max {np.max(errs):.4f}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
