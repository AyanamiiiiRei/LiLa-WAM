"""Inference policy built from a run directory. Shared by open-loop eval (step 07)
and the RMBench / XPolicyLab adapter (step 09), so both use identical preprocessing.

Interface (framework-agnostic, plain numpy):
    policy = LiLaWAMPolicy.from_run("runs/baseline/2026...", checkpoint="latest")
    policy.set_task("cover_blocks")
    policy.reset()                       # at every episode start
    policy.observe(state16, rgb)         # after EVERY env step (keeps the history right)
    chunk = policy.predict_chunk()       # (32, 14) absolute joint targets
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from . import lila_api, models
from .config import validate
from .image import to_model_input

logger = logging.getLogger(__name__)


def bspline_smooth(seq: np.ndarray, degree: int = 3, num_ctrl_pts: int = 8) -> np.ndarray:
    """Least-squares B-spline smoothing, same as the main project's robotwin_infer.bspline_smooth."""
    from scipy.interpolate import make_lsq_spline
    n = seq.shape[0]
    if n <= num_ctrl_pts:
        return seq
    x = np.arange(n)
    internal = np.linspace(0, n - 1, num_ctrl_pts - degree + 2)[1:-1]
    knots = np.concatenate([[0] * (degree + 1), internal, [n - 1] * (degree + 1)])
    return make_lsq_spline(x, seq, knots, k=degree)(x)


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    ck_dir = run_dir / "checkpoints"
    if checkpoint == "latest":
        return ck_dir / (ck_dir / "latest.txt").read_text().strip()
    p = Path(checkpoint)
    if p.is_file():
        return p
    if checkpoint.isdigit():
        return ck_dir / f"step_{int(checkpoint):07d}.pt"
    return ck_dir / checkpoint


class LiLaWAMPolicy:
    def __init__(self, cfg, checkpoint_path: str | Path, norm_stats_path: str | Path,
                 vtt_dir: str | Path | None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.vtt_dir = Path(vtt_dir) if vtt_dir else None
        built = models.build(cfg, str(norm_stats_path), device, dtype, training=False)
        self.model, self.action_model = built.wrapper, built.action_model
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.action_model.load_state_dict(ck["model_state_dict"], strict=True)
        self.model.eval().to(device, dtype)
        self.checkpoint_step = ck.get("global_step")

        self.state_offsets = list(cfg.dataset.indices_config.state_indices)
        self.image_size = tuple(OmegaConf.to_container(cfg.dataset.image_size))
        self.chunk = int(cfg.common.action_chunk_size)
        self.action_dim = int(cfg.common.action_dim)
        self.n_steps = int(cfg.common.num_inference_steps)
        self.smooth = bool(cfg.inference.get("smooth_actions", True))
        self.smooth_pts = int(cfg.inference.get("smooth_ctrl_pts", 8))
        from .schema import get_state_layout
        from .transforms import build_transforms
        self.state_transforms = build_transforms(cfg.rmbench.get("state_transforms", []),
                                                 get_state_layout(cfg.rmbench.state_layout))
        hist = 1 + max(0, -min(self.state_offsets))
        self.states: deque = deque(maxlen=hist)
        self.rgb: np.ndarray | None = None
        self.first_rgb: np.ndarray | None = None
        self.task: str | None = None
        self.task_cond: torch.Tensor | None = None
        logger.info(f"policy ready: ckpt step {self.checkpoint_step}, state offsets {self.state_offsets}")

    @classmethod
    def from_run(cls, run_dir: str | Path, checkpoint: str = "latest", device: str = "cuda",
                 dtype: torch.dtype | None = None) -> "LiLaWAMPolicy":
        run_dir = Path(run_dir)
        cfg = OmegaConf.load(run_dir / "config.yaml")
        validate(cfg)
        if dtype is None:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        vtt = run_dir / "vtt"
        return cls(cfg, resolve_checkpoint(run_dir, checkpoint), run_dir / "norm_stats.json",
                   vtt if vtt.is_dir() else None, device, dtype)

    # ----------------------------------------------------------------- episode
    def set_task(self, task: str) -> None:
        self.task = task
        if not self.cfg.model.get("use_task_cond", False):
            return
        p = self.vtt_dir / task / "task_cond.npy" if self.vtt_dir else None
        if p is None or not p.is_file():
            raise FileNotFoundError(f"No VTT for task {task!r} in {self.vtt_dir} (was it in training?)")
        self.task_cond = torch.from_numpy(np.load(p).astype(np.float32)).to(self.device, self.dtype)[None]

    def reset(self) -> None:
        self.states.clear()
        for t in self.state_transforms:
            t.reset()
        self.rgb = self.first_rgb = None

    def observe(self, state: np.ndarray, rgb: np.ndarray | None = None) -> None:
        """Record one env step. ``rgb`` = decoded HxWx3 uint8 RGB (None = keep the previous one)."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        for t in self.state_transforms:          # same transform as training, applied step by step
            state = t.step(state)
        self.states.append(state)
        if rgb is not None:
            self.rgb = rgb
            if self.first_rgb is None:
                self.first_rgb = rgb

    def _state_history(self) -> np.ndarray:
        buf = list(self.states)
        n = len(buf)
        # offset k (<=0) -> buf[n-1+k], clipped to the first observed state (= training clip at t=0)
        return np.stack([buf[max(0, n - 1 + k)] for k in self.state_offsets])

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_chunk(self, generator: torch.Generator | None = None) -> np.ndarray:
        if not self.states or self.rgb is None:
            raise RuntimeError("observe() must be called (with an image) before predict_chunk()")
        if self.cfg.model.get("use_task_cond", False) and self.task_cond is None:
            raise RuntimeError("set_task() was not called")
        state = torch.from_numpy(self._state_history()).to(self.device, self.dtype)[None]
        px = torch.from_numpy(to_model_input(self.rgb, self.image_size)).to(self.device, self.dtype)[None]

        qpos = self.model.normalize_state(state)
        feats = self.model.get_vision_features(px)
        x = torch.randn((1, self.chunk, self.action_dim), device=self.device, dtype=self.dtype,
                        generator=generator)
        ts = torch.linspace(0, 1, self.n_steps + 1, device=self.device, dtype=self.dtype)
        for i in range(self.n_steps):
            v = lila_api.predict_velocity(self.action_model, ts[i][None], x, qpos, feats, self.task_cond)
            x = x + v * (ts[i + 1] - ts[i])
        act = self.model.denormalize_action(x)[0].float().cpu().numpy()
        if self.smooth:
            act = bspline_smooth(act, 3, self.smooth_pts)
        return act.astype(np.float32)
