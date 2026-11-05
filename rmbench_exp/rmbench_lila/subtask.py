"""Sub-task design (two plans, docs/SUBTASK.md): LiLa-WAM as the executor of sub-tasks, plus a
Sub Task Chooser that decides when the current sub-task has ended and which one comes next.

Library (scripts/step36_subtask_library.py): every demo is split into sub-tasks; a sub-task
TYPE = (task, description) with its sub-task VTT = mean CLS(end) - CLS(start) of its segments.

Executor input (appended after the 64 visual tokens; earlier indices unchanged):
  sub-task token   = Linear(LN(sub-task VTT)) + type_emb[type]
  selected memory  = attention of the sub-task token over the memory entries; entry 0 =
                     (CLS of frame 0, same), entry j+1 = (CLS at the start, CLS at the end) of the
                     j-th finished sub-task; at most num_mem entries
Chooser: a small transformer over the DiT's final-norm output sequence + current CLS + the
current type embedding + a query token -> P(current sub-task has ended), next type (masked to
the task's own types).

Timing convention (same in training and at inference; no label leakage):
  at a chunk start t, the chooser sees the PRE-decision state (types/memory of sub-tasks whose
  grid end g_j < t) from a forward at flow time 0; if it decides "ended", the entry is written
  with the CLS at t, the next type is set, and the actions of this chunk are predicted with the
  POST-decision state (g_j <= t). The executor is trained on post-decision states only.

Plan 1 (joint): executor and chooser trained together; the chooser loss also trains the DiT.
Plan 2 (separate): executor trained alone (lambda_chooser 0), then frozen; the chooser is
trained on its outputs (scripts/step38_train_chooser.py).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 16


# ------------------------------------------------------------------ library / segments
class Library:
    def __init__(self, lib_dir: str | Path):
        lib_dir = Path(lib_dir)
        z = np.load(lib_dir / "library.npz", allow_pickle=False)
        self.names = [str(x) for x in z["names"]]
        self.task = [str(x) for x in z["task"]]
        self.vtt = z["vtt"].astype(np.float32)
        self.segments = json.loads((lib_dir / "segments.json").read_text())
        self.n = len(self.names)
        self.first_type: dict[str, int] = {}
        for t, eps in self.segments.items():
            firsts = [row[0][0] for row in eps.values() if row]
            self.first_type[t] = int(max(set(firsts), key=firsts.count))

    def task_mask(self, task: str) -> np.ndarray:
        return np.array([t == task for t in self.task], bool)

    def episode(self, task: str, stem: str, length: int):
        """types (n,), grid ends (n-1,) of one demo: the end of sub-task j rounded up to the
        chunk grid, strictly increasing, inside the episode (later sub-tasks dropped if not)."""
        rows = self.segments[task][stem]
        types = [r[0] for r in rows]
        last_grid = ((length - 1) // GRID) * GRID
        ends, prev = [], 0
        for j, r in enumerate(rows[:-1]):
            g = max(int(math.ceil(r[2] / GRID) * GRID), prev + GRID)
            if g > last_grid:
                types = types[: j + 1]
                break
            ends.append(g)
            prev = g
        return types, ends


def state_at(t: int, types, ends, cls: np.ndarray, num_mem: int, pre: bool):
    """Current type and memory entries at chunk/frame t. pre=True: before this chunk's decision
    (ends < t), pre=False: after it (ends <= t)."""
    done = sum((e < t) if pre else (e <= t) for e in ends)
    cur = types[min(done, len(types) - 1)]
    mem = np.zeros((num_mem, 2, cls.shape[1]), np.float32)
    mask = np.zeros(num_mem, bool)
    mem[0, 0] = mem[0, 1] = cls[0]
    mask[0] = True
    starts = [0] + list(ends)
    keep = list(range(done))[-(num_mem - 1):] if num_mem > 1 else []
    for slot, j in enumerate(keep, start=1):
        mem[slot, 0], mem[slot, 1] = cls[starts[j]], cls[ends[j]]
        mask[slot] = True
    return done, cur, mem, mask


# ------------------------------------------------------------------ modules
class SubtaskModule(nn.Module):
    def __init__(self, n_types: int, num_mem: int, cls_dim: int, hidden: int):
        super().__init__()
        self.vtt_proj = nn.Sequential(nn.LayerNorm(cls_dim), nn.Linear(cls_dim, hidden))
        self.type_emb = nn.Embedding(n_types, hidden)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.ln_first, self.ln_last = nn.LayerNorm(cls_dim), nn.LayerNorm(cls_dim)
        self.mem_mlp = nn.Sequential(nn.Linear(2 * cls_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.mem_mlp[-1].weight)
        nn.init.zeros_(self.mem_mlp[-1].bias)
        self.mem_pos = nn.Parameter(torch.randn(num_mem, hidden) * 0.02)
        self.q, self.k = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)

    def forward(self, vtt, typ, mem, mask):
        s = self.vtt_proj(vtt) + self.type_emb(typ)
        e = self.mem_mlp(torch.cat([self.ln_first(mem[:, :, 0]), self.ln_last(mem[:, :, 1])], -1)) \
            + self.mem_pos.to(s.dtype)[None]
        att = (self.q(s)[:, None] * self.k(e)).sum(-1) / math.sqrt(s.shape[-1])
        att = att.float().masked_fill(~mask, -1e4)
        w = att.softmax(-1)
        sel = (w.to(e.dtype)[..., None] * e).sum(1)
        return torch.stack([s, sel], 1), w


class Chooser(nn.Module):
    def __init__(self, hidden: int, cls_dim: int, n_types: int, d: int = 256, layers: int = 2,
                 heads: int = 4, max_len: int = 160):
        super().__init__()
        self.in_proj = nn.Linear(hidden, d)
        self.seq_pos = nn.Parameter(torch.zeros(1, max_len, d))
        self.cls_proj = nn.Sequential(nn.LayerNorm(cls_dim), nn.Linear(cls_dim, d))
        self.type_emb = nn.Embedding(n_types, d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.end_head, self.next_head = nn.Linear(d, 1), nn.Linear(d, n_types)

    def forward(self, seq, cls, typ, tmask):
        dt = self.in_proj.weight.dtype
        x = self.in_proj(seq.to(dt)) + self.seq_pos[:, : seq.shape[1]].to(dt)
        x = torch.cat([self.query.to(dt).expand(len(seq), -1, -1), x,
                       self.cls_proj(cls.to(dt))[:, None], self.type_emb(typ)[:, None]], 1)
        h = self.enc(x)[:, 0]
        end = self.end_head(h)[:, 0].float()
        nxt = self.next_head(h).float().masked_fill(~tmask, -1e4)
        return end, nxt


def attach_subtask(action_model, n_types: int, num_mem: int, cls_dim: int, chooser_cfg: dict) -> None:
    from .stage_memory import _forward_with_memory
    base = type(action_model)

    class SubtaskVLAModel(base):
        def set_subtask_input(self, vtt, typ, mem, mask):
            self._sub_in = (vtt, typ, mem, mask)

        def clear_subtask_input(self):
            self._sub_in = None

        def forward(self, t, noisy_actions, qpos_history=None, dino_features_list=None, task_cond=None):
            if getattr(self, "_sub_in", None) is None:
                raise RuntimeError("sub-task model called without set_subtask_input()")
            toks, w = self.subtask(*self._sub_in)
            self._sub_ptr = w.detach()
            out = _forward_with_memory(self, t, noisy_actions, qpos_history, dino_features_list, task_cond,
                                       memory_tokens=toks)
            return out

    SubtaskVLAModel.__name__ = f"Subtask{base.__name__}"
    action_model.__class__ = SubtaskVLAModel
    action_model.subtask = SubtaskModule(n_types, num_mem, cls_dim, action_model.hidden_dim)
    action_model.chooser = Chooser(action_model.hidden_dim, cls_dim, n_types, **chooser_cfg)
    action_model._sub_in = None


def chooser_loss(end_logit, nxt, end, nxt_label, pos_weight: float = 4.0):
    l_end = F.binary_cross_entropy_with_logits(end_logit, end.float(),
                                               pos_weight=torch.tensor(pos_weight, device=end_logit.device))
    m = end == 1
    l_next = F.cross_entropy(nxt[m], nxt_label[m]) if m.any() else end_logit.sum() * 0
    with torch.no_grad():
        pred = end_logit > 0
        stats = {"tp": int((pred & m).sum()), "fp": int((pred & ~m).sum()), "fn": int((~pred & m).sum()),
                 "next_ok": int((nxt[m].argmax(-1) == nxt_label[m]).sum()) if m.any() else 0, "n_end": int(m.sum())}
    return l_end + l_next, stats


def executor_seq(wrapper, action_model, batch, sel, grad: bool):
    """Pre-decision forward (flow time 0, noise) of the samples `sel` -> DiT final-norm sequence."""
    dev, dt = wrapper.device, wrapper.dtype
    px = batch["pixel_values"][sel]
    feats = wrapper.get_vision_features(px)
    q = batch["state"][sel].to(dev, dt)
    if q.dim() == 2:
        q = q.unsqueeze(1)
    qpos = wrapper.normalize_state(q)
    tc = batch["task_cond"][sel].to(dev, dt) if batch.get("task_cond") is not None else None
    n = int(sel.sum())
    x0 = torch.randn(tuple(batch["action_sequence"][sel].shape), device=dev, dtype=dt)
    t0 = torch.zeros(n, device=dev, dtype=dt)
    action_model.set_subtask_input(batch["sub_vtt_pre"][sel].to(dev, dt), batch["sub_type_pre"][sel].to(dev),
                                   batch["sub_mem_pre"][sel].to(dev, dt), batch["sub_mask_pre"][sel].to(dev))
    try:
        with torch.set_grad_enabled(grad):
            out = action_model(t=t0, noisy_actions=x0, qpos_history=qpos, dino_features_list=feats, task_cond=tc)
    finally:
        action_model.clear_subtask_input()
    return out["seq"] if grad else out["seq"].detach()


def run_chooser(action_model, batch, sel, seq):
    dev = seq.device
    return action_model.chooser(seq, batch["sub_cls"][sel].to(dev), batch["sub_type_pre"][sel].to(dev),
                                batch["sub_tmask"][sel].to(dev))


def chooser_pass(wrapper, action_model, batch, sel, grad: bool):
    """Pre-decision forward + chooser. grad=True: the chooser loss also trains the DiT (plan 1)."""
    seq = executor_seq(wrapper, action_model, batch, sel, grad)
    with torch.set_grad_enabled(grad):
        return run_chooser(action_model, batch, sel, seq)


def attach_subtask_wrapper(wrapper, action_model, lambda_chooser: float) -> None:
    base = type(wrapper)

    class SubtaskWrapper(base):
        def forward(self, batch):
            dev, dt = self.device, self.dtype
            action_model.set_subtask_input(batch["sub_vtt"].to(dev, dt), batch["sub_type"].to(dev),
                                           batch["sub_mem"].to(dev, dt), batch["sub_mask"].to(dev))
            try:
                loss, info = super().forward(batch)
            finally:
                action_model.clear_subtask_input()
            sel = batch["sub_end"] >= 0                     # chunk-grid anchors
            if lambda_chooser > 0 and bool(sel.any()):
                end_logit, nxt = chooser_pass(self, action_model, batch, sel, grad=True)
                lc, st = chooser_loss(end_logit, nxt, batch["sub_end"][sel].to(dev), batch["sub_next"][sel].to(dev))
                loss = loss + lambda_chooser * lc
                self._ch["loss"] += float(lc); self._ch["n"] += 1
                for k, v in st.items():
                    self._ch[k] += v
            return loss, info

        def trainer_callbacks(self, trainer):
            path = Path(trainer.run_dir) / "chooser_train.csv"

            def flush(tr, metrics):
                if not self._ch["n"]:
                    return
                c = self._ch
                new = not path.exists()
                with open(path, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["step", "loss_chooser", "end_precision", "end_recall", "next_acc", "n_end"])
                    p = c["tp"] / max(c["tp"] + c["fp"], 1)
                    r = c["tp"] / max(c["tp"] + c["fn"], 1)
                    w.writerow([metrics["step"], c["loss"] / c["n"], round(p, 3), round(r, 3),
                                round(c["next_ok"] / max(c["n_end"], 1), 3), c["n_end"]])
                self._ch.clear()
            return [flush]

    SubtaskWrapper.__name__ = f"Subtask{base.__name__}"
    wrapper.__class__ = SubtaskWrapper
    wrapper._ch = defaultdict(float)


# ------------------------------------------------------------------ inference state
class OnlineSubtask:
    def __init__(self, lib: Library, num_mem: int, cls_dim: int):
        self.lib, self.num_mem, self.cls_dim = lib, num_mem, cls_dim
        self.task = None
        self.reset()

    def set_task(self, task: str):
        if task not in self.lib.first_type:
            raise ValueError(f"task {task!r} not in the sub-task library")
        self.task = task
        self.tmask = self.lib.task_mask(task)

    def reset(self):
        self.entries: list = []
        self.start = None
        self.cur = self.lib.first_type.get(self.task, 0) if self.task else 0
        self.log: list = []
        self.chunk = 0

    def tensors(self, device, dtype):
        mem = np.zeros((self.num_mem, 2, self.cls_dim), np.float32)
        mask = np.zeros(self.num_mem, bool)
        for i, (a, b) in enumerate(self.entries[: 1] + self.entries[1:][-(self.num_mem - 1):]):
            mem[i, 0], mem[i, 1] = a, b
            mask[i] = True
        return (torch.from_numpy(self.lib.vtt[self.cur])[None].to(device, dtype),
                torch.tensor([self.cur], device=device),
                torch.from_numpy(mem)[None].to(device, dtype), torch.from_numpy(mask)[None].to(device))
