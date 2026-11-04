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
                 vtt_dir: str | Path | None, device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
                 memory_override: str | None = None):
        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.vtt_dir = Path(vtt_dir) if vtt_dir else None
        built = models.build(cfg, str(norm_stats_path), device, dtype, training=False)
        self.model, self.action_model = built.wrapper, built.action_model
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.action_model.load_state_dict(ck["model_state_dict"], strict=True)
        self.model.eval().to(device, dtype)
        self.checkpoint_step = ck.get("global_step")

        # Stage memory (only for runs with model.stage_memory.enabled; None otherwise, and then
        # nothing below changes the original behaviour). memory_override="empty" = M-empty eval.
        self.memory = None
        self.learned_switch = models.uses_learned_switch(cfg)   # phase 3: the model decides writes
        self.switch_log: list = []                              # (chunk, mean p) per decision
        if models.uses_stage_memory(cfg):
            from .stage_memory import OnlineStageMemory
            sm = cfg.model.stage_memory
            if memory_override not in (None, "empty"):
                raise ValueError(f"memory_override must be None or 'empty', got {memory_override!r}")
            self.memory = OnlineStageMemory(int(sm.num_slots), built.dino_hidden_size,
                                            str(sm.get("mode", "stage")), force_empty=memory_override == "empty")
        elif memory_override is not None:
            raise ValueError("memory_override given but this run has no stage memory")

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
                 dtype: torch.dtype | None = None, memory_override: str | None = None) -> "LiLaWAMPolicy":
        run_dir = Path(run_dir)
        cfg = OmegaConf.load(run_dir / "config.yaml")
        validate(cfg)
        if dtype is None:
            dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        vtt = run_dir / "vtt"
        return cls(cfg, resolve_checkpoint(run_dir, checkpoint), run_dir / "norm_stats.json",
                   vtt if vtt.is_dir() else None, device, dtype, memory_override)

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
        if self.memory is not None:
            self.memory.reset()
        self.switch_log = []

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
    def _inputs(self, generator):
        state = torch.from_numpy(self._state_history()).to(self.device, self.dtype)[None]
        px = torch.from_numpy(to_model_input(self.rgb, self.image_size)).to(self.device, self.dtype)[None]
        qpos = self.model.normalize_state(state)
        feats = self.model.get_vision_features(px)
        x = torch.randn((1, self.chunk, self.action_dim), device=self.device, dtype=self.dtype,
                        generator=generator)
        ts = torch.linspace(0, 1, self.n_steps + 1, device=self.device, dtype=self.dtype)
        return px, qpos, feats, x, ts

    def _foresight(self, px, qpos, feats, x, ts):
        """Phase-2 switch-head inputs with the CURRENT memory (before this chunk's write):
        last-layer CLS of the frame, and the foresight local change = mean predicted future
        patch feature (first flow step, t=0) - mean current patch feature of the same DINOv3
        layer. Also returns the first-step velocity so it can be reused if nothing is written."""
        from .stage_memory import frame_cls
        cls_now = frame_cls(self.model.vision_encoder, px, self.device, self.dtype)[0]
        cur = self.model.get_future_target_features(px).float().mean(dim=1)
        self.action_model.set_memory_input(*self.memory.tensors(self.device, self.dtype))
        try:
            out = lila_api.forward_full(self.action_model, ts[0][None], x, qpos, feats, self.task_cond)
        finally:
            self.action_model.clear_memory_input()
        delta = (lila_api.foresight_patch_mean(self.action_model, out["cond_tokens"]) - cur)[0]
        return cls_now, delta.cpu().numpy(), out["final_pred"]

    def _predict_learned(self, px, qpos, feats, x, ts) -> np.ndarray:
        """Phase 3: act with the current memory; the switch probability is averaged over the flow
        steps (training saw random flow times); a switch writes the next slot with this chunk's
        CLS, visible from the next chunk (same convention as training, learned_switch.py)."""
        from .stage_memory import frame_cls
        cls_now = frame_cls(self.model.vision_encoder, px, self.device, self.dtype)[0]
        self.memory.update(cls_now, stages_done=None)          # no external trigger
        self.action_model.set_memory_input(*self.memory.tensors(self.device, self.dtype))
        cls_t = torch.from_numpy(cls_now)[None].to(self.device)
        count = torch.tensor([self.memory.n_written], device=self.device)
        ps = []
        try:
            for i in range(self.n_steps):
                v = lila_api.predict_velocity(self.action_model, ts[i][None], x, qpos, feats, self.task_cond)
                ps.append(float(torch.sigmoid(self.action_model.switch_logit(cls_t, count))[0]))
                x = x + v * (ts[i + 1] - ts[i])
        finally:
            self.action_model.clear_memory_input()
        p = float(np.mean(ps))
        self.switch_log.append((self.memory._chunk - 1, round(p, 3)))
        if p > 0.5:
            self.memory.write_now(cls_now)
        act = self.model.denormalize_action(x)[0].float().cpu().numpy()
        if self.smooth:
            act = bspline_smooth(act, 3, self.smooth_pts)
        return act.astype(np.float32)

    @torch.no_grad()
    def stage_features(self, generator: torch.Generator | None = None):
        """(CLS, foresight change) for the observed frame with the memory as currently set
        (offline feature extraction for the switch head, scripts/step28)."""
        if self.memory is None:
            raise RuntimeError("stage_features needs a stage-memory run")
        px, qpos, feats, x, ts = self._inputs(generator)
        cls_now, delta, _ = self._foresight(px, qpos, feats, x, ts)
        return cls_now, delta

    @torch.no_grad()
    def predict_chunk(self, generator: torch.Generator | None = None, stages_done: int | None = None,
                      head=None) -> np.ndarray:
        """``stages_done``: completed-stage count from an external trigger (oracle / proprio /
        time). ``head``: a triggers.SwitchHead that decides the write itself (stage memory only)."""
        if not self.states or self.rgb is None:
            raise RuntimeError("observe() must be called (with an image) before predict_chunk()")
        if self.cfg.model.get("use_task_cond", False) and self.task_cond is None:
            raise RuntimeError("set_task() was not called")
        px, qpos, feats, x, ts = self._inputs(generator)
        if self.learned_switch:
            return self._predict_learned(px, qpos, feats, x, ts)
        v0 = None
        if self.memory is not None:
            from .stage_memory import frame_cls
            if head is not None:
                k = self.memory.n_written
                cls_now, delta, v_first = self._foresight(px, qpos, feats, x, ts)
                stages_done = head.decide(cls_now, delta, k, self.memory._chunk)
                if stages_done == k:
                    v0 = v_first                     # memory unchanged: same first step
            else:
                cls_now = frame_cls(self.model.vision_encoder, px, self.device, self.dtype)[0]
            self.memory.update(cls_now, stages_done)
            self.action_model.set_memory_input(*self.memory.tensors(self.device, self.dtype))
        try:
            for i in range(self.n_steps):
                if i == 0 and v0 is not None:
                    v = v0
                else:
                    v = lila_api.predict_velocity(self.action_model, ts[i][None], x, qpos, feats, self.task_cond)
                x = x + v * (ts[i + 1] - ts[i])
        finally:
            if self.memory is not None:
                self.action_model.clear_memory_input()
        act = self.model.denormalize_action(x)[0].float().cpu().numpy()
        if self.smooth:
            act = bspline_smooth(act, 3, self.smooth_pts)
        return act.astype(np.float32)
