"""Stage-memory phase 3, option A: the model learns WHEN to switch stages (docs/STAGE_MEMORY.md).

No boundaries, labels, rules or oracle are used anywhere in training. Same memory as M
(slot n = MLP(LN(CLS first), LN(CLS last)) + slot_emb[n]); what is learned is the switch:

  * one extra "switch token" is appended to the DiT sequence (after the memory tokens);
  * at every chunk c a small head reads that token's DiT output (DETACHED), the current
    last-layer CLS and the number of slots already written, and outputs p_c = P(switch now);
  * a switch at chunk c writes the next slot with (CLS at the stage start, CLS at chunk c);
    it is visible from chunk c+1 on (decision and use never need two forward passes);
  * training unrolls whole demos chunk by chunk (t = 0, 16, 32, ...). The forward value of a
    switch is a Bernoulli sample s_c ~ p_c (at most num_slots writes); the backward uses the
    straight-through estimate s_c + p_c - sg(p_c). The written weight of slot n at chunk c' is
    w_n = clamp(sum_{c<c'} s_c - n, 0, 1) (0 or 1 in the forward; gradient via soft_written), so the action loss of later
    chunks tells the head whether an earlier/later switch would have helped;
  * no prior on WHERE to switch. On HOW OFTEN, one of:
      switch_cost (default design, task-independent): + switch_cost * (sum_c p_c) / C per demo,
        i.e. every switch must pay for itself in action loss; the number of switches is free
        up to num_slots (slots are spare capacity, e.g. 6). The cost ramps up linearly from 0
        between switch_cost_start and switch_cost_full steps, so that the memory content is
        learned (writes happen) before switches become costly; Bernoulli sampling keeps an
        exploration floor explore_p.
      count_target (optional): + lambda_count * ((sum_c p_c - K) / K)^2 per demo.
Loss = mean over chunks of the original loss (L_fm + 0.5 L_ff) + the switch prior.
Each chunk's loss is back-propagated right away (its DiT graph is then freed); only the tiny
head graphs are kept across the demo.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .stage_memory import _forward_with_memory, attach_stage_memory

logger = logging.getLogger(__name__)


class SwitchModule(nn.Module):
    def __init__(self, hidden: int, cls_dim: int, num_slots: int, prior_rate: float, head_hidden: int = 256):
        super().__init__()
        self.num_slots = num_slots
        self.token = nn.Parameter(torch.randn(hidden) * 0.02)
        self.ln_feat = nn.LayerNorm(hidden)
        self.ln_cls = nn.LayerNorm(cls_dim)
        self.head = nn.Sequential(nn.Linear(hidden + cls_dim + num_slots + 1, head_hidden), nn.GELU(),
                                  nn.Linear(head_hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, math.log(prior_rate / (1 - prior_rate)))

    def logit(self, switch_feat: torch.Tensor, cls_now: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        oh = torch.nn.functional.one_hot(count.clamp(max=self.num_slots).long(), self.num_slots + 1)
        dt = self.head[0].weight.dtype
        x = torch.cat([self.ln_feat(switch_feat.to(dt)), self.ln_cls(cls_now.to(dt)), oh.to(dt)], dim=-1)
        return self.head(x)[:, 0].float()


def attach_learned_switch(action_model, num_slots: int, cls_dim: int, prior_rate: float) -> None:
    attach_stage_memory(action_model, num_slots, cls_dim)
    base = type(action_model)

    class LearnedSwitchVLAModel(base):
        def forward(self, t, noisy_actions, qpos_history=None, dino_features_list=None, task_cond=None):
            mem = getattr(self, "_stage_memory_input", None)
            if mem is None:
                raise RuntimeError("learned-switch model called without memory (set_memory_input())")
            tokens = self.stage_memory(*mem)
            sw = self.switch.token.to(tokens.dtype)[None, None].expand(tokens.shape[0], 1, -1)
            out = _forward_with_memory(self, t, noisy_actions, qpos_history, dino_features_list, task_cond,
                                       memory_tokens=torch.cat([tokens, sw], dim=1))
            self._switch_feat = out["tail"][:, -1].detach()        # the head never trains the DiT
            return out

        def switch_logit(self, cls_now: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
            return self.switch.logit(self._switch_feat, cls_now, count)

    LearnedSwitchVLAModel.__name__ = f"LearnedSwitch{base.__name__}"
    action_model.__class__ = LearnedSwitchVLAModel
    action_model.switch = SwitchModule(action_model.hidden_dim, cls_dim, num_slots, prior_rate)
    action_model._switch_feat = None


def soft_written(n_st: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Write weight of each slot from the straight-through switch count n_st (B,):
    forward clamp(count - n, 0, 1) (exactly 0 or 1); backward d w_n / d n_st = 1 for the slot
    just written (count - n = 1) and the next free slot (count - n = 0). torch.clamp gives
    zero gradient exactly at its bounds, which is where an integer count always sits."""
    ar = torch.arange(num_slots, device=n_st.device, dtype=n_st.dtype)[None]
    nv = n_st.detach().round()[:, None] - ar              # s + p - sg(p) is 1 only up to rounding
    gate = ((nv >= 0) & (nv <= 1)).to(n_st.dtype)
    return nv.clamp(0, 1) + gate * (n_st - n_st.detach())[:, None]


# ----------------------------------------------------------------------- training
class _PlanSampler(torch.utils.data.Sampler):
    """Yields, for every optimizer step from `start`, the dataset indices of chunk c of each of
    the step's demos (one list per chunk), so the DataLoader workers prefetch in order."""

    def __init__(self, plan, offsets, start=0):
        self.plan, self.offsets, self.start = plan, offsets, start

    def __iter__(self):
        for eps, n_chunks in self.plan[self.start:]:
            for c in range(n_chunks):
                yield [int(self.offsets[e] + c * 16) for e in eps]

    def __len__(self):
        return sum(n for _, n in self.plan[self.start:])


def _make_plan(lengths, steps, per_step, seed):
    rng = np.random.default_rng(seed)
    plan = []
    for _ in range(steps):
        eps = rng.choice(len(lengths), per_step, replace=False).tolist()
        n_chunks = int(min((lengths[e] - 1) // 16 + 1 for e in eps))
        plan.append((eps, n_chunks))
    return plan


def train(cfg, run_dir: Path, device: str, dtype, init_from: str | None) -> str:
    from . import lila_api, models
    from .data.dataset import RMBenchDataset, collate
    from .config import save_config
    from .trainer import make_scheduler
    from .utils.repro import seed_everything, worker_init_fn

    ls = cfg.model.learned_switch
    N = int(cfg.model.stage_memory.num_slots)
    steps, per_step = int(ls.total_steps), int(ls.episodes_per_step)
    target = ls.get("count_target", None)
    lam_count = float(ls.get("lambda_count", 0.0))
    sw_cost = float(ls.get("switch_cost", 0.0))
    sw_start, sw_full = int(ls.get("switch_cost_start", 0)), int(ls.get("switch_cost_full", 0))
    explore = float(ls.get("explore_p", 0.0))
    if target is None and sw_cost <= 0:
        raise ValueError("model.learned_switch needs switch_cost > 0 or a count_target")
    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    latest = run_dir / "latest.pt"
    resuming = latest.is_file()
    if resuming and init_from:
        raise ValueError(f"{run_dir} already has latest.pt; --init_from only starts NEW runs")

    ds = RMBenchDataset(cfg, "train")
    if int(cfg.dataset.get("anchor_stride", 1)) != 1:
        raise ValueError("learned switch needs dataset.anchor_stride: 1")
    offsets = np.concatenate([[0], np.cumsum(ds.lengths)[:-1]])
    cls_dir = Path(cfg.dataset.stage_memory.cls_dir)
    cls = [np.load(cls_dir / (Path(ep.path).stem + ".npy")).astype(np.float32) for ep in ds.episodes]

    if not resuming:
        save_config(cfg, run_dir / "config.yaml")
        shutil.copy2(cfg.paths.norm_stats, run_dir / "norm_stats.json")
        if cfg.model.get("use_task_cond", False):
            for task in sorted({ep.task for ep in ds.episodes}):
                (run_dir / "vtt" / task).mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(cfg.dataset.task_cond_dir) / task / "task_cond.npy", run_dir / "vtt" / task)
        ds.save_manifest(run_dir / "train_episodes.json")

    built = models.build(cfg, str(run_dir / "norm_stats.json"), device, dtype, training=True)
    wrapper, am = built.wrapper, built.action_model
    am.to(device, dtype=dtype).train()
    t = cfg.training
    opt = torch.optim.AdamW(am.parameters(), lr=float(t.learning_rate), betas=tuple(t.betas),
                            weight_decay=float(t.weight_decay))
    sched = make_scheduler(opt, steps, int(ls.get("warmup_steps", 100)), float(t.learning_rate), float(t.lr_min))
    step = 0
    if init_from and not resuming:
        ck = torch.load(init_from, map_location="cpu", weights_only=False)
        models.load_init_weights(am, ck["model_state_dict"], cfg)
        logger.info(f"initialised weights from {init_from}")
    if resuming:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        am.load_state_dict(ck["model_state_dict"], strict=True)
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        step = int(ck["global_step"])
        logger.info(f"resuming from step {step}")

    plan = _make_plan(ds.lengths, steps, per_step, seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=_PlanSampler(plan, offsets, start=step), collate_fn=collate,
        num_workers=int(cfg.system.num_workers), pin_memory=bool(cfg.system.pin_memory),
        worker_init_fn=worker_init_fn)
    it = iter(loader)
    csv_path = run_dir / "switch_train.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss_fm", "loss_ff", "loss_prior", "mean_p_sum", "mean_switches",
                                    "lr", "sec_per_step", "switch_chunks_first_demo"])
    gen = torch.Generator().manual_seed(seed + step)
    clip = float(t.grad_clip_norm)
    stop = run_dir / "STOP"
    tic = time.time()
    logger.info(f"learned switch: {steps} steps x {per_step} demos (whole demos, chunk by chunk); {N} slots; "
                f"prior: " + (f"count target {target} (lambda {lam_count})" if target is not None else
                              f"switch cost {sw_cost} (ramp {sw_start}->{sw_full} steps), explore {explore}")
                + f"; stop safely with: touch {stop}")
    ac = dict(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=dtype != torch.float32)
    while step < steps:
        eps, C = plan[step]
        B = len(eps)
        opt.zero_grad(set_to_none=True)
        pairs = torch.zeros(B, N, 2, cls[0].shape[1])
        count = torch.zeros(B, dtype=torch.long)
        n_st = torch.zeros(B, device=device)
        p_sum = torch.zeros(B, device=device)
        start = [cls[e][0] for e in eps]
        prev = [cls[e][0] for e in eps]
        fm = ff = 0.0
        sw_chunks: list[list[int]] = [[] for _ in range(B)]
        for c in range(C):
            batch = next(it)
            cand = pairs.clone()                    # next slot: what a switch at c-1 would have written
            for b in range(B):
                if count[b] < N:
                    cand[b, count[b], 0] = torch.from_numpy(start[b])
                    cand[b, count[b], 1] = torch.from_numpy(prev[b])
            w = soft_written(n_st, N)
            am.set_memory_input(cand.to(device, dtype), w)
            with torch.autocast(**ac):
                loss_c, info = wrapper(batch)
            (loss_c / C).backward(retain_graph=True)
            fm += float(info.get("loss_mse", 0.0)) / C
            ff += float(info.get("loss_future_feat", 0.0)) / C
            now = [cls[e][min(c * 16, len(cls[e]) - 1)] for e in eps]
            with torch.autocast(**ac):
                logit = am.switch_logit(torch.from_numpy(np.stack(now)).to(device), count.to(device))
            p = torch.sigmoid(logit)
            p_sample = p.detach().cpu().clamp(min=explore)
            s = torch.bernoulli(p_sample, generator=gen).to(device)
            s = s * (count.to(device) < N).float()
            n_st = n_st + s + p - p.detach()
            p_sum = p_sum + p
            for b in range(B):
                if s[b] > 0:
                    pairs[b, count[b], 0] = torch.from_numpy(start[b])
                    pairs[b, count[b], 1] = torch.from_numpy(now[b])
                    start[b] = now[b]
                    count[b] += 1
                    sw_chunks[b].append(c)
            prev = now
            am.clear_memory_input()
            del loss_c, info
        if target is not None:
            loss_count = lam_count * (((p_sum - float(target)) / float(target)) ** 2).mean()
        else:
            ramp = 1.0 if sw_full <= sw_start else min(max((step - sw_start) / (sw_full - sw_start), 0.0), 1.0)
            loss_count = sw_cost * ramp * (p_sum / C).mean()
        loss_count.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(), clip)
        opt.step()
        sched.step()
        step += 1
        dt = time.time() - tic
        tic = time.time()
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([step, round(fm, 5), round(ff, 5), round(float(loss_count), 5),
                                    round(float(p_sum.mean()), 3), round(float(count.float().mean()), 3),
                                    f"{opt.param_groups[0]['lr']:.2e}", round(dt, 2),
                                    json.dumps(sw_chunks[0])])
        if step % int(ls.get("log_every", 10)) == 0 or step == 1:
            logger.info(f"step {step}/{steps} fm {fm:.4f} ff {ff:.4f} prior {float(loss_count):.5f} "
                        f"sum p {float(p_sum.mean()):.2f} switches {float(count.float().mean()):.2f} "
                        f"(demo {eps[0]}: chunks {sw_chunks[0]}) {dt:.1f}s/step "
                        f"ETA {dt * (steps - step) / 3600:.1f}h")
        done = step == steps
        if step % int(ls.get("save_every", 250)) == 0 or done:
            path = run_dir / "checkpoints" / f"step_{step:07d}.pt"
            torch.save(lila_api.checkpoint_payload(am, global_step=step,
                                                   config=OmegaConf.to_container(cfg, resolve=True)), path)
            (run_dir / "checkpoints" / "latest.txt").write_text(path.name)
            torch.save({"model_state_dict": am.state_dict(), "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(), "global_step": step}, latest)
            logger.info(f"saved {path}")
        if stop.exists() and not done:
            stop.unlink()
            torch.save({"model_state_dict": am.state_dict(), "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(), "global_step": step}, latest)
            logger.info(f"[paused] stopped at step {step}; run the same command again to resume")
            return "stopped"
    return "finished"
