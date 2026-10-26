"""Finding and reading RMBench episodes (h5py + numpy only, no torch).

Directory layout produced by RMBench's downloader:
    <data_root>/<task>/<embodiment>/data/episode_XXXXXXX.hdf5
    <data_root>/<task>/<embodiment>/video/episode_XXXXXXX.mp4
where <data_root> is ``.../RMBench/data/demo_clean``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np

from .codec import decode_image_bit
from .schema import (ACTION_ALIGNMENTS, IMAGE_KEY, STATE_GROUP, VISION_GROUP,
                     VectorLayout, camera_h5_name)


@dataclass(frozen=True)
class EpisodeRef:
    task: str
    path: str
    episode_id: int

    @property
    def video_path(self) -> Path:
        p = Path(self.path)
        return p.parent.parent / "video" / (p.stem + ".mp4")


def _episode_id(p: Path) -> int:
    digits = "".join(ch for ch in p.stem if ch.isdigit())
    return int(digits) if digits else -1


def scan_episodes(data_root: str | Path, embodiment: str = "aloha_agilex",
                  tasks: Iterable[str] | None = None) -> list[EpisodeRef]:
    root = Path(data_root)
    if (root / "demo_clean").is_dir():          # tolerate pointing at RMBench/data
        root = root / "demo_clean"
    if not root.is_dir():
        raise FileNotFoundError(f"RMBench data root not found: {root}")
    wanted = set(tasks) if tasks else None
    refs: list[EpisodeRef] = []
    for task_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if wanted is not None and task_dir.name not in wanted:
            continue
        data_dir = task_dir / embodiment / "data"
        for h5 in sorted(data_dir.glob("*.hdf5"), key=_episode_id):
            refs.append(EpisodeRef(task_dir.name, str(h5), _episode_id(Path(h5))))
    if wanted:
        missing = wanted - {r.task for r in refs}
        if missing:
            raise FileNotFoundError(f"No episodes for tasks {sorted(missing)} under {root}/<task>/{embodiment}/data")
    if not refs:
        raise FileNotFoundError(f"No episodes under {root}/<task>/{embodiment}/data/*.hdf5")
    return refs


def split_episodes(refs: Sequence[EpisodeRef], holdout_per_task: int,
                   max_train_per_task: int | None = None) -> tuple[list[EpisodeRef], list[EpisodeRef]]:
    """Deterministic split: the last ``holdout_per_task`` episodes (by id) of each task are held out."""
    by_task: dict[str, list[EpisodeRef]] = {}
    for r in refs:
        by_task.setdefault(r.task, []).append(r)
    train, held = [], []
    for task in sorted(by_task):
        eps = sorted(by_task[task], key=lambda r: r.episode_id)
        k = min(holdout_per_task, max(len(eps) - 1, 0))
        tr, ho = (eps[:-k], eps[-k:]) if k > 0 else (eps, [])
        if max_train_per_task is not None:
            tr = tr[:max_train_per_task]
        train += tr
        held += ho
    return train, held


def select_episodes(cfg, split: str) -> list[EpisodeRef]:
    """Episodes of a config's split: 'train' | 'holdout' | 'all'."""
    r = cfg.rmbench
    refs = scan_episodes(r.data_root, r.embodiment, r.get("tasks", None))
    if split == "all":
        return refs
    train, held = split_episodes(refs, int(r.get("holdout_per_task", 0) or 0),
                                 r.get("max_episodes_per_task", None))
    if split not in ("train", "holdout"):
        raise ValueError(f"unknown split {split!r}")
    return train if split == "train" else held


class EpisodeReader:
    """Context manager around one HDF5 episode."""

    def __init__(self, path: str):
        self.path = path
        self.f: h5py.File | None = None

    def __enter__(self) -> "EpisodeReader":
        self.f = h5py.File(self.path, "r")
        return self

    def __exit__(self, *exc) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None

    # ------------------------------------------------------------------ sizes
    def length(self, layout: VectorLayout, group: str = STATE_GROUP) -> int:
        return int(self.f[group][layout.fields[0].h5_key].shape[0])

    # ---------------------------------------------------------------- vectors
    def read_vector(self, layout: VectorLayout, group: str, idx: np.ndarray | None = None) -> np.ndarray:
        """Concatenate ``layout`` fields of ``group`` at time indices ``idx`` (None = all)."""
        parts = []
        for field in layout.fields:
            ds = self.f[group][field.h5_key]
            if idx is None:
                arr = ds[()]
            else:
                uniq, inv = np.unique(np.asarray(idx), return_inverse=True)  # h5py needs sorted, unique
                arr = ds[uniq][inv]
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.shape[1] != field.dim:
                raise ValueError(f"{self.path}: {group}/{field.h5_key} has dim {arr.shape[1]}, "
                                 f"layout {layout.name!r} expects {field.dim}")
            parts.append(arr)
        return np.concatenate(parts, axis=1)

    def read_action(self, layout: VectorLayout, alignment: str, idx: np.ndarray | None = None) -> np.ndarray:
        return self.read_vector(layout, ACTION_ALIGNMENTS[alignment], idx)

    # ----------------------------------------------------------------- images
    def num_frames(self, camera: str) -> int:
        return int(self.f[VISION_GROUP][camera_h5_name(camera)][IMAGE_KEY].shape[0])

    def raw_frame(self, camera: str, i: int):
        return self.f[VISION_GROUP][camera_h5_name(camera)][IMAGE_KEY][int(i)]

    def read_frames(self, camera: str, idx: Sequence[int]) -> np.ndarray:
        """Decoded RGB uint8 frames, shape (len(idx), H, W, 3). No colour conversion afterwards!"""
        ds = self.f[VISION_GROUP][camera_h5_name(camera)][IMAGE_KEY]
        cache: dict[int, np.ndarray] = {}
        out = []
        for i in idx:
            i = int(i)
            if i not in cache:
                cache[i] = decode_image_bit(ds[i])
            out.append(cache[i])
        return np.stack(out, axis=0)

    def has_camera(self, camera: str) -> bool:
        return VISION_GROUP in self.f and camera_h5_name(camera) in self.f[VISION_GROUP]

    def frequency(self) -> int | None:
        try:
            return int(np.asarray(self.f["additional_info"]["frequency"][()]))
        except KeyError:
            return None


def clip_indices(anchor: int, offsets: Sequence[int], length: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute indices clipped into [0, length-1] and a mask of which were in range."""
    raw = anchor + np.asarray(offsets, dtype=np.int64)
    valid = (raw >= 0) & (raw < length)
    return np.clip(raw, 0, length - 1), valid
