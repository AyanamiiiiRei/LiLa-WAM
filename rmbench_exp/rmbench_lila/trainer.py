"""Step-based trainer for LiLa-WAM on RMBench (plan steps 6–8).

Differences from the main project's train.py (which stays untouched):
  * training length and LR schedule are defined in OPTIMIZER steps (data size differs a
    lot between RoboTwin and RMBench; steps make runs comparable)
  * seeds for python/numpy/torch and DataLoader workers
  * the logged loss is the true per-sample loss (no grad_accum scaling artefact),
    split into flow-matching and foresight parts
  * every run directory is self-contained: resolved config, norm stats, VTTs, episode
    manifest, git commits, loss CSV, checkpoints -> eval needs only the run directory
  * checkpoints keep the main project's format ('model_state_dict'), so the main
    project's robotwin_infer.py can load them too
"""

from __future__ import annotations

import csv
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Callable, Iterator

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import lila_api, models
from .config import save_config
from .data import build_dataset, collate
from .utils.repro import seed_everything, worker_init_fn, write_run_info

logger = logging.getLogger(__name__)


def make_scheduler(optimizer, total_steps: int, warmup: int, lr: float, lr_min: float) -> LambdaLR:
    """Linear warm-up then cosine from lr to lr_min over total_steps optimizer steps."""
    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return (lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * p))) / lr
    return LambdaLR(optimizer, factor)


class Trainer:
    def __init__(self, cfg: DictConfig, run_dir: Path, device: str, dtype: torch.dtype,
                 init_from: str | None = None, resume: str | None = None,
                 fixed_batch: bool = False):
        self.cfg, self.run_dir, self.device, self.dtype = cfg, Path(run_dir), device, dtype
        self.fixed_batch = fixed_batch
        self.callbacks: list[Callable[["Trainer", dict], None]] = []   # extension hook
        t = cfg.training

        seed_everything(int(cfg.experiment.seed))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)

        # ---- data
        self.dataset = build_dataset(cfg, "train")
        g = torch.Generator()
        g.manual_seed(int(cfg.experiment.seed))
        self.loader = DataLoader(
            self.dataset, batch_size=int(t.batch_size), shuffle=True, drop_last=True,
            num_workers=int(cfg.system.num_workers), pin_memory=bool(cfg.system.pin_memory),
            collate_fn=collate, worker_init_fn=worker_init_fn, generator=g,
            persistent_workers=int(cfg.system.num_workers) > 0,
        )
        if len(self.loader) == 0:
            raise ValueError("Fewer samples than one batch; lower training.batch_size")

        # ---- self-contained run directory
        save_config(cfg, self.run_dir / "config.yaml")
        shutil.copy2(cfg.paths.norm_stats, self.run_dir / "norm_stats.json")
        if cfg.model.get("use_task_cond", False):
            vtt_dst = self.run_dir / "vtt"
            for task in sorted({ep.task for ep in self.dataset.episodes}):
                (vtt_dst / task).mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(cfg.dataset.task_cond_dir) / task / "task_cond.npy", vtt_dst / task)
        self.dataset.save_manifest(self.run_dir / "train_episodes.json")
        write_run_info(self.run_dir, {"dataset": self.dataset.describe(),
                                      "init_from": init_from, "resume": resume})

        # ---- model
        built = models.build(cfg, str(self.run_dir / "norm_stats.json"), device, dtype, training=True)
        self.model, self.action_model = built.wrapper, built.action_model
        self.action_model.to(device, dtype=dtype).train()
        n_train = sum(p.numel() for p in self.action_model.parameters() if p.requires_grad)
        logger.info(f"trainable parameters: {n_train / 1e6:.1f}M")

        self.optimizer = AdamW(self.action_model.parameters(), lr=float(t.learning_rate),
                               betas=tuple(t.betas), weight_decay=float(t.weight_decay))
        self.total_steps = int(t.total_steps)
        self.scheduler = make_scheduler(self.optimizer, self.total_steps, int(t.get("warmup_steps", 0)),
                                        float(t.learning_rate), float(t.lr_min))
        self.global_step = 0
        self.samples_seen = 0

        if init_from:
            ck = torch.load(init_from, map_location="cpu", weights_only=False)
            self.action_model.load_state_dict(ck["model_state_dict"], strict=True)
            logger.info(f"initialised weights from {init_from} (fresh optimizer/schedule)")
        if resume:
            ck = torch.load(resume, map_location="cpu", weights_only=False)
            self.action_model.load_state_dict(ck["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            self.scheduler.load_state_dict(ck["scheduler_state_dict"])
            self.global_step = int(ck["global_step"])
            self.samples_seen = int(ck.get("samples_seen", 0))
            logger.info(f"resumed from {resume} at step {self.global_step}")

        self.csv_path = self.run_dir / "loss.csv"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["step", "epoch", "loss", "loss_fm", "loss_ff", "lr", "sec_per_step"])

    # ------------------------------------------------------------------ data
    def _batches(self) -> Iterator[dict]:
        if self.fixed_batch:
            batch = next(iter(self.loader))
            while True:
                yield batch
        while True:
            yield from self.loader

    @property
    def epoch(self) -> float:
        return self.samples_seen / len(self.dataset)

    # ------------------------------------------------------------------ loop
    def train(self) -> Path:
        t = self.cfg.training
        accum, clip = int(t.grad_accum_steps), float(t.grad_clip_norm)
        log_every, save_every = int(t.log_every_steps), int(t.save_every_steps)
        lam = float(t.get("lambda_future_feat", 0.0))
        batches = self._batches()
        logger.info(f"training {self.total_steps} optimizer steps, effective batch "
                    f"{int(t.batch_size) * accum}, {len(self.dataset)} samples "
                    f"(~{self.total_steps * int(t.batch_size) * accum / len(self.dataset):.1f} epochs)")
        tic = time.time()
        while self.global_step < self.total_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss_sum = fm_sum = ff_sum = 0.0
            for _ in range(accum):
                batch = next(batches)
                with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu",
                                    dtype=self.dtype, enabled=self.dtype != torch.float32):
                    loss, info = self.model(batch)
                (loss / accum).backward()
                loss_sum += loss.item() / accum
                fm_sum += float(info.get("loss_mse", 0.0)) / accum
                ff_sum += float(info.get("loss_future_feat", 0.0)) / accum
                self.samples_seen += int(t.batch_size)
            torch.nn.utils.clip_grad_norm_(self.action_model.parameters(), clip)
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            if self.global_step % log_every == 0 or self.global_step == 1:
                dt = (time.time() - tic) / (log_every if self.global_step > 1 else 1)
                tic = time.time()
                metrics = {"step": self.global_step, "epoch": round(self.epoch, 3), "loss": loss_sum,
                           "loss_fm": fm_sum, "loss_ff": ff_sum,
                           "lr": self.optimizer.param_groups[0]["lr"], "sec_per_step": dt}
                with open(self.csv_path, "a", newline="") as f:
                    csv.writer(f).writerow([metrics[k] for k in
                                            ("step", "epoch", "loss", "loss_fm", "loss_ff", "lr", "sec_per_step")])
                logger.info(f"step {self.global_step}/{self.total_steps} ep {self.epoch:.2f} "
                            f"loss {loss_sum:.4f} (fm {fm_sum:.4f} + {lam}*ff {ff_sum:.4f}) "
                            f"lr {metrics['lr']:.2e} {dt:.2f}s/step")
                for cb in self.callbacks:
                    cb(self, metrics)

            if self.global_step % save_every == 0 or self.global_step == self.total_steps:
                self.save()
        return self.run_dir

    def save(self) -> Path:
        path = self.run_dir / "checkpoints" / f"step_{self.global_step:07d}.pt"
        payload = lila_api.checkpoint_payload(
            self.action_model, self.optimizer, self.scheduler,
            global_step=self.global_step, samples_seen=self.samples_seen, epoch=self.epoch,
            config=OmegaConf.to_container(self.cfg, resolve=True))
        torch.save(payload, path)
        (self.run_dir / "checkpoints" / "latest.txt").write_text(path.name)
        logger.info(f"saved {path}")
        return path
