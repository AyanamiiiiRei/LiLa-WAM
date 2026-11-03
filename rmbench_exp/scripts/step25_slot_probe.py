#!/usr/bin/env python
"""Stage-memory experiment: open-loop probe of the decision after move 2 vs after move 3.

  python scripts/step25_slot_probe.py --run runs/sm_M-seed0 [--compare runs/sm_A-seed0]

Question: at the two moments where the image cannot tell "do move 3" from "press the
button" (both arms home, two trays occupied), does the M model's choice follow slot 3?

For every demo (45 train + 5 held-out) with stage ends e1 < e2 < e3 (boundaries_gt.json):
  t = e2  (move 2 just done; the demo then does move 3)
  t = e3  (move 3 just done; the demo then presses)
the model predicts an action chunk from the TRUE frame and state at t, with slots 1-2 written
as in training and slot 3 either EMPTY or WRITTEN (first = CLS at e2, last = CLS at t).
Both demo continuations start from the same pose (arms home), so each prediction is labelled
"press" if it is closer (joint L2 over the chunk) to the demo's press chunk actions[e3:e3+32]
than to its move-3 chunk actions[e2:e2+32]. Several noise seeds per case.

Reading the result (fraction labelled "press"):
  correct use of slot 3:  t=e2/empty -> ~0,   t=e3/written -> ~1
  slot 3 ignored:         same fraction with empty and written
  jitter hypothesis:      t=e3/empty also -> high (trained: "slot 3 empty, still press")
--compare runs a memoryless model (A) on the same frames as a reference (no slots).
"""

from __future__ import annotations

from pathlib import Path

import _common
import numpy as np
import torch
from omegaconf import OmegaConf

from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.policy import LiLaWAMPolicy
from rmbench_lila.schema import STATE_GROUP, get_action_layout, get_state_layout
from rmbench_lila.stages import load_boundaries


def predict(policy, state, rgb, slots, seed):
    policy.reset()
    policy.observe(state, rgb)
    if policy.memory is not None:
        m = policy.memory
        m.cls[:] = 0
        m.written[:] = False
        for k, (first, last) in slots.items():
            m.cls[k, 0], m.cls[k, 1] = first, last
            m.written[k] = True
        m.stage_start = np.zeros(m.cls_dim, np.float32)     # non-None: update() then writes nothing
    g = torch.Generator(device=policy.device).manual_seed(seed)
    return policy.predict_chunk(g, stages_done=None)


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", required=True)
    p.add_argument("--compare", default=None, help="memoryless run evaluated on the same frames")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    bounds = load_boundaries(cfg.dataset.stage_memory.boundaries)
    cls_dir = Path(cfg.dataset.stage_memory.cls_dir)
    cam = list(cfg.dataset.camera_names)[0]
    s_layout = get_state_layout(cfg.rmbench.state_layout)
    a_layout = get_action_layout(cfg.rmbench.action_layout)
    held = {r.path for r in select_episodes(cfg, "holdout")}
    refs = select_episodes(cfg, "all")

    models = {"M": LiLaWAMPolicy.from_run(run, "latest", args.device)}
    if args.compare:
        models["A"] = LiLaWAMPolicy.from_run(Path(args.compare), "latest", args.device)
    for m in models.values():
        m.set_task("swap_blocks")          # loads the task's VTT (task_cond)

    counts: dict = {}
    for ref in refs:
        name = Path(ref.path).name
        e1, e2, e3 = [s[1] for s in bounds[name][:3]]
        cls = np.load(cls_dir / (Path(ref.path).stem + ".npy")).astype(np.float32)
        with EpisodeReader(ref.path) as r:
            n = r.length(s_layout, STATE_GROUP)
            states = r.read_vector(s_layout, STATE_GROUP)
            acts = r.read_action(a_layout, cfg.rmbench.action_alignment)
            if e3 + 32 > n:
                continue
            move_ref, press_ref = acts[e2:e2 + 32], acts[e3:e3 + 32]
            frames = r.read_frames(cam, [e2, e3])
        split = "heldout" if ref.path in held else "train"
        for t, rgb in ((e2, frames[0]), (e3, frames[1])):
            tname = "after move 2 (t=e2)" if t == e2 else "after move 3 (t=e3)"
            base = {0: (cls[0], cls[e1]), 1: (cls[e1], cls[e2])}
            cases = {"M slot3 empty": base, "M slot3 written": {**base, 2: (cls[e2], cls[t])}}
            for label, slots in cases.items():
                for s in range(args.seeds):
                    a = predict(models["M"], states[t], rgb, slots, s)
                    is_press = np.linalg.norm(a - press_ref) < np.linalg.norm(a - move_ref)
                    counts.setdefault((tname, label, split), []).append(is_press)
            if "A" in models:
                for s in range(args.seeds):
                    a = predict(models["A"], states[t], rgb, {}, s)
                    is_press = np.linalg.norm(a - press_ref) < np.linalg.norm(a - move_ref)
                    counts.setdefault((tname, "A (no memory)", split), []).append(is_press)

    print("\nfraction of predictions closer to the demo's PRESS chunk than to its MOVE-3 chunk")
    print(f"{'moment':22s} {'model / memory':18s} {'train':>14s} {'held-out':>14s}")
    for tname in ("after move 2 (t=e2)", "after move 3 (t=e3)"):
        for label in ("M slot3 empty", "M slot3 written", "A (no memory)"):
            row = []
            for split in ("train", "heldout"):
                v = counts.get((tname, label, split))
                row.append(f"{np.mean(v):.2f} (n={len(v)})" if v else "-")
            if any(x != "-" for x in row):
                print(f"{tname:22s} {label:18s} {row[0]:>14s} {row[1]:>14s}")
    print("\nexpected if slot 3 is used: after move 2 / empty ~0; after move 3 / written ~1")


if __name__ == "__main__":
    main()
