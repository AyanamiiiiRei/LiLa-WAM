"""Dataset features. Each feature turns one (episode, anchor time) into tensors.

Output keys follow what the main project's VLAWrapper.forward expects:
    state (T_s, 16), action_sequence (32, 14), pixel_values (3, H, W),
    future_pixel_values (3, H, W), task_cond (D,)
Extra keys (e.g. first_pixel_values) are ignored by the main model and are meant
for models extended inside this sub-project.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from ..episodes import EpisodeReader, EpisodeRef, clip_indices
from ..image import to_model_input
from ..schema import STATE_GROUP, get_action_layout, get_state_layout
from .registry import FEATURES


@dataclass
class SampleContext:
    reader: EpisodeReader
    ref: EpisodeRef
    anchor: int          # current time step t (0-based, < length)
    length: int          # number of (state, action, frame) steps in the episode
    train: bool


class Feature:
    registry_name: str = ""

    def __init__(self, cfg: DictConfig, train: bool):
        self.cfg = cfg
        self.train = train

    def __call__(self, ctx: SampleContext) -> dict:
        raise NotImplementedError


def _primary_camera(cfg: DictConfig) -> str:
    return list(cfg.dataset.camera_names)[0]


def _image_size(cfg: DictConfig) -> tuple[int, int]:
    w, h = cfg.dataset.image_size
    return int(w), int(h)


@FEATURES.register("state")
class StateFeature(Feature):
    """Proprioception at ``state_indices`` offsets (clipped to the episode start)."""

    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        self.layout = get_state_layout(cfg.rmbench.state_layout)
        self.offsets = list(cfg.dataset.indices_config.state_indices)

    def __call__(self, ctx):
        idx, valid = clip_indices(ctx.anchor, self.offsets, ctx.length)
        vec = ctx.reader.read_vector(self.layout, STATE_GROUP, idx)
        return {"state": torch.from_numpy(vec), "state_mask": torch.from_numpy(valid)}


@FEATURES.register("action_chunk")
class ActionChunkFeature(Feature):
    """Absolute joint targets at ``action_indices``; past the end the last target repeats."""

    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        self.layout = get_action_layout(cfg.rmbench.action_layout)
        self.alignment = cfg.rmbench.action_alignment
        self.offsets = list(cfg.dataset.indices_config.action_indices)

    def __call__(self, ctx):
        idx, valid = clip_indices(ctx.anchor, self.offsets, ctx.length)
        vec = ctx.reader.read_action(self.layout, self.alignment, idx)
        return {"action_sequence": torch.from_numpy(vec), "action_mask": torch.from_numpy(valid)}


class _ImageFeatureBase(Feature):
    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        self.camera = _primary_camera(cfg)
        self.size = _image_size(cfg)
        self.aug_ops = []
        if train and cfg.dataset.get("image_aug", False):
            import torchvision.transforms as T
            self.aug_ops = [T.ColorJitter(brightness=0.05), T.ColorJitter(contrast=0.05),
                            T.ColorJitter(saturation=0.05), T.ColorJitter(hue=0.05)]

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if not self.aug_ops:
            return img
        from PIL import Image
        pil = Image.fromarray(img)
        for op in random.sample(self.aug_ops, random.choice([1, 2])):
            pil = op(pil)
        return np.asarray(pil)

    def _encode(self, frames: np.ndarray, augment: bool) -> torch.Tensor:
        arr = np.stack([to_model_input(self._augment(f) if augment else f, self.size) for f in frames])
        t = torch.from_numpy(arr)
        return t[0] if t.shape[0] == 1 else t      # (3,H,W) for a single frame, else (T,3,H,W)


@FEATURES.register("image")
class ImageFeature(_ImageFeatureBase):
    """Current frame(s) of the primary camera at ``camera_indices`` offsets."""

    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        self.offsets = list(cfg.dataset.indices_config.camera_indices)

    def __call__(self, ctx):
        idx, _ = clip_indices(ctx.anchor, self.offsets, ctx.length)
        frames = ctx.reader.read_frames(self.camera, idx)
        return {"pixel_values": self._encode(frames, augment=True)}


@FEATURES.register("future_image")
class FutureImageFeature(_ImageFeatureBase):
    """Frame at anchor + future_frame_offset (default = chunk size); no augmentation (it is a target)."""

    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        off = cfg.dataset.get("future_frame_offset", None)
        self.offset = int(off) if off is not None else int(cfg.common.action_chunk_size)

    def __call__(self, ctx):
        idx, _ = clip_indices(ctx.anchor, [self.offset], ctx.length)
        frames = ctx.reader.read_frames(self.camera, idx)
        return {"future_pixel_values": self._encode(frames, augment=False)}


@FEATURES.register("first_image")
class FirstImageFeature(_ImageFeatureBase):
    """Frame 0 of the episode (control experiment B2). Not consumed by the unmodified model."""

    def __call__(self, ctx):
        frames = ctx.reader.read_frames(self.camera, [0])
        return {"first_pixel_values": self._encode(frames, augment=True)}


@FEATURES.register("task_cond")
class TaskCondFeature(Feature):
    """Precomputed VTT vector for the episode's task (step 05 output)."""

    def __init__(self, cfg, train):
        super().__init__(cfg, train)
        self.root = Path(cfg.dataset.task_cond_dir)
        self.cache: dict[str, torch.Tensor] = {}

    def preload(self, tasks):
        for t in sorted(set(tasks)):
            p = self.root / t / "task_cond.npy"
            if not p.is_file():
                raise FileNotFoundError(f"VTT missing for task {t!r}: {p} (run scripts/step05_compute_vtt.py)")
            self.cache[t] = torch.from_numpy(np.load(p).astype(np.float32))

    def __call__(self, ctx):
        return {"task_cond": self.cache[ctx.ref.task]}
