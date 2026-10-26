#!/usr/bin/env python
"""Plan step 5: Visual Transition Tokens for the RMBench tasks (+ task similarity).

Usage:
  python scripts/step05_compute_vtt.py --config configs/exp/baseline.yaml
Output (dataset.task_cond_dir): <task>/task_cond.npy, <task>/episode_diffs.npz,
        cosine_similarity.csv, meta.json (incl. the most similar task pairs)
"""

from __future__ import annotations

from pathlib import Path

import _common
from omegaconf import OmegaConf
from tqdm import tqdm

from rmbench_lila.episodes import select_episodes
from rmbench_lila.lila_api import load_vision_encoder
from rmbench_lila.vtt import compute_and_save


def main():
    p = _common.parser(__doc__)
    p.add_argument("--split", default="train", choices=["train", "all"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="defaults to dataset.task_cond_dir")
    args = p.parse_args()
    cfg = _common.load(args)
    device, dtype = _common.pick_device_dtype(args.device)

    refs = select_episodes(cfg, args.split)
    model, hidden = load_vision_encoder(cfg, device, dtype)
    out = Path(args.out or cfg.dataset.task_cond_dir)
    meta = compute_and_save(
        model, refs, list(cfg.dataset.camera_names)[0],
        tuple(OmegaConf.to_container(cfg.dataset.image_size)), device, dtype, out,
        meta_extra={"dino_checkpoint": cfg.model.vision_encoder.checkpoint_path, "hidden_size": hidden,
                    "split": args.split, "data_root": cfg.rmbench.data_root},
        progress=lambda x: tqdm(x, desc="VTT"),
    )
    print(f"\nSaved VTTs for {len(meta['tasks'])} tasks to {out}")
    s = meta["similarity"]
    print(f"off-diagonal cosine: mean {s['mean_offdiag']:.3f}, max {s['max_offdiag']:.3f}")
    for a, b, c in s["most_similar_pairs"]:
        print(f"  {a} <-> {b}: {c}")


if __name__ == "__main__":
    main()
