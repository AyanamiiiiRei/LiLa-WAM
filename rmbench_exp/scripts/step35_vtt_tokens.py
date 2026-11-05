#!/usr/bin/env python
"""Stage-memory phase 3 (representation analysis): the DiT output at the VTT position, per chunk.

  python scripts/step35_vtt_tokens.py --runs runs/sm_A-seed0 runs/sm_M-seed0 runs/sm_S-seed0

The INPUT VTT is one constant vector for swap_blocks (single-task training), but its DiT
OUTPUT (sequence position 35: actions 0-31, state 32, registers 33-34, VTT 35) changes with
the frame, the state and (for M/S) the memory. Question: does this token form stage-specific
representations? For every demo (45 train + 5 held-out) and every chunk start t = 0, 16, ...,
the model is run along the demo (true frames/states) exactly as in the closed loop:
  A  no memory
  M  memory written at the oracle-equivalent move ends (boundaries_gt.json; the way M runs)
  S  learned switch, its own decisions
and the token = final-norm output at position 35, averaged over the flow-matching steps of
that chunk's prediction (fixed noise seed per chunk). Also saved: current-frame CLS (cache),
chunk index, true stage (boundaries_gt.json; for EVALUATION only, never used to extract).
Output: artifacts/stage_memory/<task>/vtt_tokens/<run name>/<episode>.npz
Analysis: analysis/vtt_token_analysis.py
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
from rmbench_lila.stages import GRID, load_boundaries, stage_index


def vtt_position(cfg) -> int:
    m = cfg.model
    if not m.get("use_task_cond", False):
        raise SystemExit("run has no VTT token (model.use_task_cond is false)")
    return int(cfg.common.action_chunk_size) + int(cfg.common.proprio_len) + int(m.get("num_registers", 0))


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--runs", nargs="+", default=["runs/sm_A-seed0", "runs/sm_M-seed0", "runs/sm_S-seed0"])
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    art = Path(f"artifacts/stage_memory/{args.task}")
    bounds = load_boundaries(art / "boundaries_gt.json")
    cls_dir = art / "cls"

    for run_s in args.runs:
        run = Path(run_s)
        cfg = OmegaConf.load(run / "config.yaml")
        pos = vtt_position(cfg)
        cam = list(cfg.dataset.camera_names)[0]
        s_layout = get_state_layout(cfg.rmbench.state_layout)
        held = {r.path for r in select_episodes(cfg, "holdout")}
        pol = LiLaWAMPolicy.from_run(run, "latest", args.device)
        pol.set_task(args.task)
        mode = "learned" if pol.learned_switch else ("oracle" if pol.memory is not None else "none")
        grabbed: list = []
        hook = pol.action_model.final_norm.register_forward_hook(
            lambda mod, inp, out: grabbed.append(out[0, pos].float().cpu().numpy()))
        out_dir = art / "vtt_tokens" / run.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{run.name}: memory mode {mode}, VTT position {pos}")
        try:
            for ref in select_episodes(cfg, "all"):
                name = Path(ref.path).name
                stages = bounds[name]
                ends = [s[1] for s in stages[:-1]]
                cls = np.load(cls_dir / (Path(ref.path).stem + ".npy")).astype(np.float32)
                with EpisodeReader(ref.path) as r:
                    n = r.length(s_layout, STATE_GROUP)
                    states = r.read_vector(s_layout, STATE_GROUP)
                    ts = list(range(0, n, GRID))
                    frames = r.read_frames(cam, ts)
                pol.reset()
                toks, writes = [], []
                for t, rgb in zip(ts, frames):
                    pol.observe(states[t], rgb)
                    grabbed.clear()
                    g = torch.Generator(device=pol.device).manual_seed(1000 + t)
                    if mode == "oracle":
                        pol.predict_chunk(g, stages_done=sum(t >= e for e in ends))
                    else:
                        pol.predict_chunk(g)
                    lay = getattr(pol.action_model, "_last_layout", None)
                    if lay is not None and lay["dino_start"] - 1 != pos:
                        raise RuntimeError(f"VTT position mismatch: {pos} vs layout {lay}")
                    toks.append(np.mean(grabbed, axis=0))
                    writes.append(int(pol.memory.n_written) if pol.memory is not None else 0)
                np.savez_compressed(out_dir / (Path(ref.path).stem + ".npz"), tok=np.array(toks, np.float32),
                                    cls=cls[ts], t=np.array(ts), stage=np.array([stage_index(t, stages) for t in ts]),
                                    written=np.array(writes), split="heldout" if ref.path in held else "train")
                print(f"  {name}: {len(ts)} chunks, slots written at the end: {writes[-1]}")
        finally:
            hook.remove()
        del pol
        torch.cuda.empty_cache()
    print("done; next: python analysis/vtt_token_analysis.py")


if __name__ == "__main__":
    main()
