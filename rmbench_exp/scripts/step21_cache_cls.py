#!/usr/bin/env python
"""Stage-memory experiment, step 2: cache the frozen DINOv3 last-layer CLS of every frame.

  python scripts/step21_cache_cls.py --task swap_blocks
  -> artifacts/stage_memory/<task>/cls/<episode>.npy   (T, 1024) float16

The memory tokens are built from these vectors during training. The CLS is exactly what
the policy computes online (rmbench_lila.stage_memory.frame_cls: same preprocessing
image.to_model_input, same encoder output last_hidden_state[:, 0], same dtype), so the
train/inference consistency check (step 23) can compare them directly.
"""

from __future__ import annotations

from pathlib import Path

import _common
import numpy as np
import torch
from tqdm import tqdm

from rmbench_lila.episodes import EpisodeReader, scan_episodes
from rmbench_lila.image import to_model_input
from rmbench_lila.lila_api import load_vision_encoder
from rmbench_lila.stage_memory import frame_cls


def main():
    p = _common.parser(__doc__)
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--out", default=None)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = _common.load(args)
    device, dtype = _common.pick_device_dtype(args.device)
    model, _ = load_vision_encoder(cfg, device, dtype)
    cam = list(cfg.dataset.camera_names)[0]
    size = tuple(cfg.dataset.image_size)
    out = Path(args.out or f"artifacts/stage_memory/{args.task}/cls")
    out.mkdir(parents=True, exist_ok=True)

    refs = scan_episodes(cfg.rmbench.data_root, cfg.rmbench.embodiment, [args.task])
    for ref in tqdm(refs, desc="CLS"):
        dst = out / (Path(ref.path).stem + ".npy")
        if dst.exists():
            continue
        with EpisodeReader(ref.path) as r:
            n = r.num_frames(cam)
            chunks = []
            for s in range(0, n, args.batch):
                frames = r.read_frames(cam, range(s, min(n, s + args.batch)))
                px = torch.from_numpy(np.stack([to_model_input(f, size) for f in frames]))
                chunks.append(frame_cls(model, px, device, dtype))
        arr = np.concatenate(chunks).astype(np.float16)
        np.save(dst, arr)
    print(f"cached CLS for {len(refs)} episodes in {out}")


if __name__ == "__main__":
    main()
