"""RMBench dataset for LiLa-WAM, assembled from registered features.

Unlike the main project's loader, a broken sample raises instead of being silently
replaced by a random one: with only ~50 demos per task, a silent data bug would be
far more costly than a crash.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ..episodes import EpisodeReader, EpisodeRef, select_episodes
from ..schema import STATE_GROUP, get_state_layout
from . import features as _features  # noqa: F401  (registers built-in features)
from .registry import FEATURES

logger = logging.getLogger(__name__)


class RMBenchDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "train", episodes: list[EpisodeRef] | None = None):
        self.cfg = cfg
        self.split = split
        self.train = split == "train"
        self.episodes = episodes if episodes is not None else select_episodes(cfg, split)
        if not self.episodes:
            raise ValueError(f"No episodes in split {split!r}")

        self.features = [FEATURES.get(n)(cfg, self.train) for n in cfg.dataset.features]
        for f in self.features:
            if hasattr(f, "preload"):
                f.preload(ep.task for ep in self.episodes)

        state_layout = get_state_layout(cfg.rmbench.state_layout)
        stride = int(cfg.dataset.get("anchor_stride", 1))
        self.lengths = np.array([self._length(ep.path, state_layout) for ep in self.episodes], dtype=np.int64)
        # anchors: every `stride`-th step of every episode
        self.index: list[tuple[int, int]] = [
            (e, t) for e, n in enumerate(self.lengths) for t in range(0, int(n), stride)
        ]
        tasks = sorted({ep.task for ep in self.episodes})
        logger.info(f"RMBenchDataset[{split}]: {len(self.episodes)} episodes, {len(tasks)} tasks, "
                    f"{len(self.index)} samples, features={list(cfg.dataset.features)}")

    @staticmethod
    def _length(path: str, layout) -> int:
        with h5py.File(path, "r") as f:
            return int(f[STATE_GROUP][layout.fields[0].h5_key].shape[0])

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ref = self.episodes[e]
        out: dict = {}
        with EpisodeReader(ref.path) as reader:
            ctx = _features.SampleContext(reader, ref, t, int(self.lengths[e]), self.train)
            for feat in self.features:
                for k, v in feat(ctx).items():
                    if k in out:
                        raise KeyError(f"feature {feat.registry_name!r} overwrote key {k!r}")
                    out[k] = v
        out["task_name"] = ref.task
        out["episode_id"] = ref.episode_id
        out["anchor"] = t
        return out

    def describe(self) -> dict:
        per_task: dict[str, dict] = {}
        for ep, n in zip(self.episodes, self.lengths):
            d = per_task.setdefault(ep.task, {"episodes": 0, "frames": 0})
            d["episodes"] += 1
            d["frames"] += int(n)
        return {"split": self.split, "episodes": len(self.episodes), "samples": len(self.index),
                "per_task": per_task}

    def save_manifest(self, path: Path) -> None:
        """Which episodes went into this split (kept with every run for reproducibility)."""
        rows = [{"task": ep.task, "episode_id": ep.episode_id, "path": ep.path, "length": int(n)}
                for ep, n in zip(self.episodes, self.lengths)]
        Path(path).write_text(json.dumps(rows, indent=1))


def collate(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def build_dataset(cfg: DictConfig, split: str = "train") -> RMBenchDataset:
    return RMBenchDataset(cfg, split)
