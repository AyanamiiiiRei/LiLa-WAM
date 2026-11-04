#!/usr/bin/env python
"""Stage-memory phase 3: where does the learned switch fire on the demos? (offline, no simulator)

  python scripts/step33_switch_offline.py --run runs/sm_S-seed0 [--checkpoint latest]

The learned-switch model is run along each demo (true frames and states at t = 0, 16, ...),
with its OWN memory decisions (as in the closed loop). For each demo: the chunks where it
switched, compared with the move ends (boundaries_gt.json; used ONLY here, for evaluation)
with the same metrics as analysis/stage_discovery.py: hits within +-1 chunk, ARI, and the
"mean time" baseline (training demos' mean ends). Also: switches per demo, and the spread of
each switch's chunk across demos vs the spread of the true ends (a pure timer would place
its switches at nearly constant chunks). Output: <run>/switch_offline.json and .md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import _common
import numpy as np
import torch
from omegaconf import OmegaConf

from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.policy import LiLaWAMPolicy
from rmbench_lila.schema import STATE_GROUP, get_state_layout
from rmbench_lila.stages import GRID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from stage_discovery import ari, hits, labels_from_cuts  # noqa: E402


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", default="runs/sm_S-seed0")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    bounds = json.loads(Path(f"artifacts/stage_memory/{args.task}/boundaries_gt.json").read_text())
    cam = list(cfg.dataset.camera_names)[0]
    s_layout = get_state_layout(cfg.rmbench.state_layout)
    held = {r.path for r in select_episodes(cfg, "holdout")}
    pol = LiLaWAMPolicy.from_run(run, args.checkpoint, args.device)
    if not pol.learned_switch:
        raise SystemExit("not a learned-switch run")
    pol.set_task(args.task)
    N = pol.memory.num_slots
    rows = []
    for ref in select_episodes(cfg, "all"):
        name = Path(ref.path).name
        with EpisodeReader(ref.path) as r:
            n = r.length(s_layout, STATE_GROUP)
            states = r.read_vector(s_layout, STATE_GROUP)
            ts = list(range(0, n, GRID))
            frames = r.read_frames(cam, ts)
        pol.reset()
        g = torch.Generator(device=pol.device).manual_seed(0)
        for t, rgb in zip(ts, frames):
            pol.observe(states[t], rgb)
            pol.predict_chunk(g)
        sw = [c for c, _ in pol.memory.log]
        true = [bounds[name][k][1] // GRID for k in range(N)]
        rows.append({"episode": name, "split": "heldout" if ref.path in held else "train", "n_chunks": len(ts),
                     "switches": sw, "true": true, "p": [p for _, p in pol.switch_log]})
        print(f"{name}: switches at chunks {sw}, move ends at chunks {true}")
    (run / "switch_offline.json").write_text(json.dumps(rows))

    train = [r for r in rows if r["split"] == "train"]
    mean_t = [int(round(x)) for x in np.mean([r["true"] for r in train], axis=0)]
    lines = [f"# Learned switch on the demos ({run.name}, checkpoint {pol.checkpoint_step})", "",
             "| | hits train | hits held-out | ARI train | ARI held-out |", "|---|---|---|---|---|"]
    for name, fn in (("learned switch", lambda r: r["switches"]), ("mean time (uses labels)", lambda r: mean_t)):
        cells = []
        for split in ("train", "heldout"):
            rs = [r for r in rows if r["split"] == split]
            h = sum(hits(sorted(fn(r)), r["true"]) for r in rs)
            a = np.mean([ari(labels_from_cuts(sorted(fn(r)), r["n_chunks"]), labels_from_cuts(r["true"], r["n_chunks"]))
                         for r in rs])
            cells += [f"{h}/{N * len(rs)}", f"{a:.2f}"]
        lines.append(f"| {name} | {cells[0]} | {cells[2]} | {cells[1]} | {cells[3]} |")
    nsw = [len(r["switches"]) for r in rows]
    lines += ["", f"switches per demo: " + ", ".join(f"{k}: {nsw.count(k)}" for k in range(N + 1)), "",
              "spread across demos (chunks, std): learned switch k vs true end k"]
    for k in range(N):
        sk = [r["switches"][k] for r in rows if len(r["switches"]) > k]
        tk = [r["true"][k] for r in rows if len(r["switches"]) > k]
        if len(sk) > 2:
            corr = float(np.corrcoef(sk, tk)[0, 1]) if np.std(sk) > 0 else float("nan")
            lines.append(f"- switch {k + 1}: n={len(sk)}, median chunk {int(np.median(sk))} (true {int(np.median(tk))}), "
                         f"std {np.std(sk):.2f} (true {np.std(tk):.2f}), corr with true end {corr:.2f}")
    (run / "switch_offline.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
