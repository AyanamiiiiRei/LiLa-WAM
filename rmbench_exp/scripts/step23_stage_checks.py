#!/usr/bin/env python
"""Stage-memory experiment: checks that need the GPU and real data.

  # (before training) train/inference memory consistency on one held-out demo:
  python scripts/step23_stage_checks.py consistency --config configs/stage_memory/M.yaml

  # (after training) held-out flow-matching loss per stage, any group (A, F, M):
  python scripts/step23_stage_checks.py heldout --run runs/sm_M-seed0 [--checkpoint 5000]
  python scripts/step23_stage_checks.py heldout --run runs/sm_M-seed0 --memory empty   # M-empty

consistency: at every chunk start (t = 0, 16, ...) of a held-out demo, builds the memory the
  way the POLICY does (live DINOv3 CLS of the frame, slot written when the stage has ended)
  and the way TRAINING does (dataset feature, cached CLS, no jitter) and compares them.
  Pass = same written slots everywhere and cosine(CLS) > 0.999 (bf16 rounding only).
heldout: L_fm on the 5 held-out demos (every 4th frame), split by stage (boundaries_gt.json),
  with fixed noise/time per sample so groups are comparable. Written to
  <run>/heldout_by_stage[_<tag>].csv. Used to pick the checkpoint (stage 4 = press the
  button, the step that needs memory most) and for the plan's per-stage L_fm report.
"""

from __future__ import annotations

import csv
from pathlib import Path

import _common
import numpy as np
import torch
from omegaconf import OmegaConf


def consistency(args):
    from rmbench_lila.data.stage_features import StageMemoryFeature
    from rmbench_lila.data.features import SampleContext
    from rmbench_lila.episodes import EpisodeReader, select_episodes
    from rmbench_lila.image import to_model_input
    from rmbench_lila.lila_api import load_vision_encoder
    from rmbench_lila.stage_memory import OnlineStageMemory, frame_cls
    from rmbench_lila.stages import load_boundaries

    cfg = _common.load(args)
    device, dtype = _common.pick_device_dtype(args.device)
    enc, hidden = load_vision_encoder(cfg, device, dtype)
    feat = StageMemoryFeature(cfg, train=False)
    sm = cfg.model.stage_memory
    ref = select_episodes(cfg, "holdout")[0]
    stages = load_boundaries(cfg.dataset.stage_memory.boundaries)[Path(ref.path).name]
    ends = [s[1] for s in stages[: int(sm.num_slots)]]
    mem = OnlineStageMemory(int(sm.num_slots), hidden, str(sm.get("mode", "stage")))
    cam, size = list(cfg.dataset.camera_names)[0], tuple(cfg.dataset.image_size)
    worst_cos, mismatch = 1.0, 0
    with EpisodeReader(ref.path) as r:
        n = r.num_frames(cam)
        for t in range(0, n, 16):
            px = torch.from_numpy(to_model_input(r.read_frames(cam, [t])[0], size))[None]
            mem.update(frame_cls(enc, px, device, dtype)[0], stages_done=sum(t >= e for e in ends))
            f = feat(SampleContext(r, ref, t, n, False))
            w = f["stage_mem_written"].numpy()
            if not (w == mem.written).all():
                mismatch += 1
                print(f"t={t}: written train {w} vs policy {mem.written}")
                continue
            for k in np.where(w)[0]:
                for j in (0, 1):
                    a, b = f["stage_mem_cls"][k, j].numpy(), mem.cls[k, j]
                    worst_cos = min(worst_cos, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))
    ok = mismatch == 0 and worst_cos > 0.999
    print(f"episode {Path(ref.path).name}: stage ends {ends}; written-mask mismatches {mismatch}; "
          f"worst cosine(train CLS, policy CLS) {worst_cos:.5f} -> {'PASS' if ok else 'FAIL'}")


def heldout(args):
    from rmbench_lila import models
    from rmbench_lila.data.dataset import RMBenchDataset, collate
    from rmbench_lila.episodes import select_episodes
    from rmbench_lila.policy import resolve_checkpoint
    from rmbench_lila.stages import load_boundaries, stage_index

    run = Path(args.run)
    cfg = OmegaConf.load(run / "config.yaml")
    device, dtype = _common.pick_device_dtype(args.device)
    bpath = cfg.dataset.get("stage_memory", {}).get("boundaries") or args.boundaries
    bounds = load_boundaries(bpath)
    refs = select_episodes(cfg, "holdout")
    OmegaConf.update(cfg, "dataset.anchor_stride", 4, merge=True)
    ds = RMBenchDataset(cfg, "holdout", episodes=refs)
    built = models.build(cfg, str(run / "norm_stats.json"), device, dtype, training=True)
    ck_path = resolve_checkpoint(run, args.checkpoint)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    built.action_model.load_state_dict(ck["model_state_dict"], strict=True)
    built.wrapper.eval().to(device, dtype)
    sums: dict[int, list] = {}
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=4)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if "stage_mem_written" in batch and args.memory == "empty":
                batch["stage_mem_written"] = torch.zeros_like(batch["stage_mem_written"])
            torch.manual_seed(1000 + bi)                       # same noise and time for every group
            with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
                _, info = built.wrapper(batch)
            per = ((info["pred_v"].float() - info["target_v"].float()) ** 2).mean(dim=(1, 2)).cpu().numpy()
            for i, v in enumerate(per):
                ep = Path(refs[0].path).parent / f"episode_{int(batch['episode_id'][i]):07d}.hdf5"
                s = stage_index(int(batch["anchor"][i]), bounds[ep.name])
                sums.setdefault(s, []).append(float(v))
    tag = f"_{args.memory}" if args.memory != "normal" else ""
    out = run / f"heldout_by_stage{tag}.csv"
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["checkpoint_step", "stage", "loss_fm", "n"])
        for s in sorted(sums):
            w.writerow([ck.get("global_step"), s + 1, float(np.mean(sums[s])), len(sums[s])])
            print(f"step {ck.get('global_step')} stage {s + 1}: L_fm {np.mean(sums[s]):.5f} (n={len(sums[s])})")
    print(f"appended to {out}")


def main():
    p = _common.parser(__doc__)
    p.add_argument("check", choices=["consistency", "heldout"])
    p.add_argument("--run", default=None)
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--memory", choices=["normal", "empty"], default="normal")
    p.add_argument("--boundaries", default="artifacts/stage_memory/swap_blocks/boundaries_gt.json",
                   help="for group A runs, whose config has no stage_memory section")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.check == "consistency":
        consistency(args)
    else:
        if not args.run:
            raise SystemExit("--run is required for heldout")
        heldout(args)


if __name__ == "__main__":
    main()
