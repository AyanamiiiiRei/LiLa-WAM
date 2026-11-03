"""Stage memory slots for LiLa-WAM (stage-memory experiment, docs/STAGE_MEMORY.md).

When stage n has ended:   m_n = MLP([LN(CLS_first), LN(CLS_last)]) + slot_emb[n]
Not yet ended:            m_n = empty + slot_emb[n]
DiT sequence:             [the original 100 tokens, m_1, ..., m_N]

Everything here lives in the sub-project. The main-project VLAModel / VLAWrapper are
extended by swapping the instance's class to a subclass defined below (no main-project
file changes). With ``model.stage_memory.enabled: false`` nothing in this module is used.

The memory is passed to VLAModel.forward through ``self._stage_memory_input`` because the
main project's loss function calls ``model(t, noisy_actions=..., ...)`` with a fixed
signature. It is set right before and cleared right after each call.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- CLS
@torch.no_grad()
def frame_cls(vision_encoder, pixel_values: torch.Tensor, device, dtype) -> np.ndarray:
    """Last-layer CLS (after the encoder's final norm), (B, D) float32. Used by BOTH the
    training cache (step 21) and the online policy, so the memory content is identical."""
    out = vision_encoder(pixel_values=pixel_values.to(device, dtype), return_dict=True)
    return out.last_hidden_state[:, 0, :].float().cpu().numpy()


# ------------------------------------------------------------------------ module
class StageMemory(nn.Module):
    def __init__(self, num_slots: int, cls_dim: int, hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.ln_first = nn.LayerNorm(cls_dim)
        self.ln_last = nn.LayerNorm(cls_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * cls_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        nn.init.zeros_(self.mlp[-1].weight)          # at init the content does not change the output
        nn.init.zeros_(self.mlp[-1].bias)
        self.slot_emb = nn.Parameter(torch.randn(num_slots, hidden_dim) * 0.02)
        self.empty = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, cls_pairs: torch.Tensor, written: torch.Tensor) -> torch.Tensor:
        """cls_pairs (B, N, 2, D) = (first, last) CLS per slot; written (B, N) bool -> (B, N, H)."""
        x = torch.cat([self.ln_first(cls_pairs[:, :, 0]), self.ln_last(cls_pairs[:, :, 1])], dim=-1)
        content = self.mlp(x)
        empty = self.empty.to(content.dtype).expand_as(content)
        tok = torch.where(written[..., None].bool(), content, empty)
        return tok + self.slot_emb.to(content.dtype)[None]


# ---------------------------------------------------------------- model subclass
def _forward_with_memory(self, t, noisy_actions, qpos_history=None, dino_features_list=None,
                         task_cond=None, memory_tokens=None):
    """Copy of VLAModel.forward (LiLa-WAM @ 3cfd9abd, models/vla_model_fm.py) with the memory
    tokens appended AFTER the visual tokens and the foresight decoder's "visual" slice made
    explicit (dino_start:dino_end) so it never includes the memory tokens.
    With memory_tokens=None it computes exactly the original forward (tests/test_stage_memory.py)."""
    t_emb = self.time_mlp(t)
    x_action = self.action_proj(noisy_actions) + self.action_pos_emb[:, :noisy_actions.shape[1], :] + self.type_emb_action
    x_proprio_real = self.proprio_proj(qpos_history) + self.proprio_pos_emb[:, :qpos_history.shape[1], :] + self.type_emb_proprio
    x_proprio = torch.zeros_like(x_proprio_real)
    tokens_list = [x_action, x_proprio]
    proprio_start = x_action.shape[1]
    proprio_end = proprio_start + x_proprio.shape[1]
    if self.num_registers > 0:
        tokens_list.append(self.register_tokens.expand(noisy_actions.shape[0], -1, -1))
    if self.use_task_cond:
        assert task_cond is not None, "use_task_cond=True but task_cond was not passed to forward"
        tokens_list.append(self.task_cond_proj(task_cond).unsqueeze(1) + self.type_emb_task_cond)
    dino_start = sum(x.shape[1] for x in tokens_list)
    if dino_features_list is not None:
        assert len(dino_features_list) == self.num_dino_layers
        if self.fusion_mode == "concat":
            fused = self.concat_fusion(dino_features_list)
            tokens_list.append(self.dino_adapters[0](fused) + self.type_emb_dino_layers[0])
        else:
            for layer_idx, feats in enumerate(dino_features_list):
                tokens_list.append(self.dino_adapters[layer_idx](feats) + self.type_emb_dino_layers[layer_idx])
    dino_end = sum(x.shape[1] for x in tokens_list)
    if memory_tokens is not None:
        tokens_list.append(memory_tokens.to(x_action.dtype))
    x = torch.cat(tokens_list, dim=1)

    state_injected = False
    for i, block in enumerate(self.blocks):
        if i == self.state_inject_start and not state_injected:
            x[:, proprio_start:proprio_end, :] = x_proprio_real
            state_injected = True
        x = block(x, t_emb)

    x = self.final_norm(x)
    final_pred = self.output_proj(x[:, :self.action_len, :])
    if getattr(self, "future_cond_tokens", "all") == "visual":
        cond_tokens = x[:, dino_start:dino_end, :]
    else:
        cond_tokens = x[:, self.action_len:, :]
    self._last_layout = {"dino_start": dino_start, "dino_end": dino_end, "seq_len": x.shape[1]}
    return {"final_pred": final_pred, "cond_tokens": cond_tokens}


def attach_stage_memory(action_model, num_slots: int, cls_dim: int) -> None:
    """Turn a built VLAModel into a stage-memory model (adds `stage_memory.*` parameters)."""
    base = type(action_model)

    class StageMemoryVLAModel(base):
        def forward(self, t, noisy_actions, qpos_history=None, dino_features_list=None, task_cond=None):
            mem = getattr(self, "_stage_memory_input", None)
            if mem is None:
                raise RuntimeError("stage-memory model called without memory "
                                   "(set_memory_input(); use written=False for an empty memory)")
            tokens = self.stage_memory(*mem)
            return _forward_with_memory(self, t, noisy_actions, qpos_history, dino_features_list,
                                        task_cond, memory_tokens=tokens)

        def set_memory_input(self, cls_pairs, written) -> None:
            self._stage_memory_input = (cls_pairs, written)

        def clear_memory_input(self) -> None:
            self._stage_memory_input = None

    StageMemoryVLAModel.__name__ = f"StageMemory{base.__name__}"
    action_model.__class__ = StageMemoryVLAModel
    action_model.stage_memory = StageMemory(num_slots, cls_dim, action_model.hidden_dim)
    action_model._stage_memory_input = None


# -------------------------------------------------------------- wrapper subclass
def attach_wrapper(wrapper, action_model) -> None:
    """VLAWrapper.forward(batch) passes batch['stage_mem_cls'/'stage_mem_written'] to the
    model and accumulates the flow-matching loss per stage (batch['stage_id'])."""
    base = type(wrapper)

    class StageMemoryWrapper(base):
        def forward(self, batch):
            dev, dt = self.device, self.dtype
            action_model.set_memory_input(batch["stage_mem_cls"].to(dev, dt), batch["stage_mem_written"].to(dev))
            try:
                loss, info = super().forward(batch)
            finally:
                action_model.clear_memory_input()
            with torch.no_grad():
                per = ((info["pred_v"].float() - info["target_v"].float()) ** 2).mean(dim=(1, 2)).cpu()
            for s, v in zip(batch["stage_id"].tolist(), per.tolist()):
                self._stage_fm[int(s)][0] += v
                self._stage_fm[int(s)][1] += 1
            return loss, info

        def trainer_callbacks(self, trainer):
            path = Path(trainer.run_dir) / "loss_by_stage.csv"

            def flush(tr, metrics):
                if not self._stage_fm:
                    return
                new = not path.exists()
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["step", "stage", "loss_fm", "n"])
                    for s in sorted(self._stage_fm):
                        tot, n = self._stage_fm[s]
                        w.writerow([metrics["step"], s, tot / max(n, 1), n])
                self._stage_fm.clear()
            return [flush]

    StageMemoryWrapper.__name__ = f"StageMemory{base.__name__}"
    wrapper.__class__ = StageMemoryWrapper
    wrapper._stage_fm = defaultdict(lambda: [0.0, 0])


# ------------------------------------------------------------- online memory
class OnlineStageMemory:
    """Inference-time slot state, updated once per action chunk (policy.predict_chunk).

    mode "stage":       slot n <- (CLS at the first chunk of stage n, CLS at the chunk where
                        the oracle first reports stage n done); the current frame then
                        becomes the first frame of stage n+1.
    mode "first_frame": one slot, written at the first chunk with (CLS_0, CLS_0).
    force_empty:        never write (M-empty evaluation of an M checkpoint).
    """

    def __init__(self, num_slots: int, cls_dim: int, mode: str = "stage", force_empty: bool = False):
        if mode not in ("stage", "first_frame"):
            raise ValueError(f"stage_memory.mode must be 'stage' or 'first_frame', got {mode!r}")
        self.num_slots, self.cls_dim, self.mode, self.force_empty = num_slots, cls_dim, mode, force_empty
        self.reset()

    def reset(self) -> None:
        self.cls = np.zeros((self.num_slots, 2, self.cls_dim), np.float32)
        self.written = np.zeros(self.num_slots, bool)
        self.stage_start: np.ndarray | None = None
        self.n_written = 0
        self.log: list[tuple] = []          # (chunk index, slot) writes, for the episode record
        self._chunk = 0

    def update(self, cur_cls: np.ndarray, stages_done: int | None) -> None:
        cur = np.asarray(cur_cls, np.float32).reshape(-1)
        if self.stage_start is None:
            self.stage_start = cur
            if self.mode == "first_frame" and not self.force_empty:
                self.cls[0, 0] = self.cls[0, 1] = cur
                self.written[0] = True
        if self.mode == "stage" and not self.force_empty and stages_done is not None:
            while self.n_written < min(int(stages_done), self.num_slots):
                n = self.n_written
                self.cls[n, 0], self.cls[n, 1] = self.stage_start, cur
                self.written[n] = True
                self.log.append((self._chunk, n))
                self.stage_start = cur
                self.n_written += 1
        self._chunk += 1

    def tensors(self, device, dtype):
        return (torch.from_numpy(self.cls)[None].to(device, dtype), torch.from_numpy(self.written)[None].to(device))
