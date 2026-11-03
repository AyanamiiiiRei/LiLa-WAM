"""Dataset feature for the stage-memory experiment (docs/STAGE_MEMORY.md).

Config:
  model.stage_memory:   {enabled, num_slots, mode: stage | first_frame}
  dataset.stage_memory: {boundaries: <boundaries_gt.json>, cls_dir: <step 21 output>, jitter_p: 0.5}

Output per sample (anchor frame t):
  stage_mem_cls     (N, 2, D) float32  (first, last) CLS of every slot (zeros if not written)
  stage_mem_written (N,) bool          slot n is written iff its stage ended at or before t
  stage_id          ()  int64          stage that t belongs to (for the per-stage loss log)

Never uses a stage that ends after t (no future leakage; tests/test_stage_memory.py).
Training jitters each written-slot boundary by +-1 chunk with probability jitter_p;
the held-out split is never jittered.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from ..stages import jitter_ends, load_boundaries, slots_at, stage_index
from .features import Feature
from .registry import FEATURES


@FEATURES.register("stage_memory")
class StageMemoryFeature(Feature):
    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        sm, dsm = cfg.model.stage_memory, cfg.dataset.stage_memory
        self.mode = str(sm.get("mode", "stage"))
        self.n = int(sm.num_slots)
        # loaded in both modes: "first_frame" uses them only for the per-stage loss log (stage_id)
        self.boundaries = load_boundaries(dsm.boundaries)
        self.cls_dir = Path(dsm.cls_dir)
        self.jitter_p = float(dsm.get("jitter_p", 0.0)) if train else 0.0
        self._cls: dict[str, np.ndarray] = {}
        if self.mode == "first_frame" and self.n != 1:
            raise ValueError("stage_memory.mode=first_frame uses exactly one slot (num_slots: 1)")

    def _episode_cls(self, path: str) -> np.ndarray:
        a = self._cls.get(path)
        if a is None:
            f = self.cls_dir / (Path(path).stem + ".npy")
            if not f.is_file():
                raise FileNotFoundError(f"CLS cache missing: {f} (run scripts/step21_cache_cls.py)")
            a = self._cls[path] = np.load(f).astype(np.float32)
        return a

    def __call__(self, ctx):
        cls = self._episode_cls(ctx.ref.path)
        if len(cls) != ctx.length:
            raise ValueError(f"{ctx.ref.path}: CLS cache has {len(cls)} frames, episode {ctx.length}")
        t = ctx.anchor
        out = np.zeros((self.n, 2, cls.shape[1]), np.float32)
        if self.mode == "first_frame":
            out[0, 0] = out[0, 1] = cls[0]
            written = np.ones(1, bool)
            sid = stage_index(t, self.boundaries[Path(ctx.ref.path).name])
        else:
            stages = self.boundaries[Path(ctx.ref.path).name]
            ends = jitter_ends(stages, self.n, ctx.length, self.jitter_p, rng=random)
            pairs, written = slots_at(t, ends, self.n)
            for k, (a, b) in enumerate(pairs):
                if written[k]:
                    out[k, 0], out[k, 1] = cls[a], cls[b]
            sid = stage_index(t, stages)
        return {"stage_mem_cls": torch.from_numpy(out), "stage_mem_written": torch.from_numpy(written),
                "stage_id": torch.tensor(sid, dtype=torch.int64)}
