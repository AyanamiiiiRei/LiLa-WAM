"""Dataset feature "subtask" (rmbench_lila/subtask.py, docs/SUBTASK.md).

Config: dataset.subtask: {library_dir, cls_root}; model.subtask: {num_mem}
Output per sample (anchor t):
  post-decision state (executor input; sub-tasks with grid end <= t are finished):
    sub_vtt (D,), sub_type (), sub_mem (M, 2, D), sub_mask (M,)
  pre-decision state + chooser labels (only meaningful on the chunk grid, t % 16 == 0):
    sub_vtt_pre, sub_type_pre, sub_mem_pre, sub_mask_pre, sub_cls (D,) = CLS of frame t,
    sub_end () = 1 if the current sub-task ends at t, 0 if not, -1 off the grid,
    sub_next () = type that follows (when sub_end == 1), else -100,
  sub_tmask (K,) = the types of this sample's task (the chooser chooses among them).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..subtask import GRID, Library, state_at
from .features import Feature
from .registry import FEATURES


@FEATURES.register("subtask")
class SubtaskFeature(Feature):
    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        self.lib = Library(cfg.dataset.subtask.library_dir)
        self.cls_root = Path(cfg.dataset.subtask.cls_root)
        self.num_mem = int(cfg.model.subtask.num_mem)
        self._cls: dict = {}
        self._masks: dict = {}

    def _episode_cls(self, ref):
        a = self._cls.get(ref.path)
        if a is None:
            f = self.cls_root / ref.task / "cls" / (Path(ref.path).stem + ".npy")
            if not f.is_file():
                raise FileNotFoundError(f"CLS cache missing: {f} (scripts/step21_cache_cls.py --task {ref.task})")
            a = self._cls[ref.path] = np.load(f).astype(np.float32)
        return a

    def __call__(self, ctx):
        cls = self._episode_cls(ctx.ref)
        if len(cls) != ctx.length:
            raise ValueError(f"{ctx.ref.path}: CLS cache has {len(cls)} frames, episode {ctx.length}")
        task, t = ctx.ref.task, ctx.anchor
        types, ends = self.lib.episode(task, Path(ctx.ref.path).stem, ctx.length)
        _, cur, mem, mask = state_at(t, types, ends, cls, self.num_mem, pre=False)
        done_p, cur_p, mem_p, mask_p = state_at(t, types, ends, cls, self.num_mem, pre=True)
        if t % GRID == 0:
            end = int(done_p < len(ends) and ends[done_p] == t)
            nxt = types[done_p + 1] if end else -100
        else:
            end, nxt = -1, -100
        tm = self._masks.get(task)
        if tm is None:
            tm = self._masks[task] = torch.from_numpy(self.lib.task_mask(task))
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))  # noqa: E731
        return {"sub_vtt": f32(self.lib.vtt[cur]), "sub_type": torch.tensor(cur), "sub_mem": f32(mem),
                "sub_mask": torch.from_numpy(mask), "sub_vtt_pre": f32(self.lib.vtt[cur_p]),
                "sub_type_pre": torch.tensor(cur_p), "sub_mem_pre": f32(mem_p), "sub_mask_pre": torch.from_numpy(mask_p),
                "sub_cls": f32(cls[t]), "sub_end": torch.tensor(end), "sub_next": torch.tensor(nxt),
                "sub_tmask": tm}
