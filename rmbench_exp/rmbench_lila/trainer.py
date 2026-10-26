"""Step-based, resumable trainer for LiLa-WAM on RMBench (plan steps 6–8).

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

Resuming (same mechanism as the user's train_resumable.py for RoboTwin):
  * every ``resume_every_steps`` (default 1000) an atomic ``latest.pt`` is written
    (temp file + rename, so a power cut never leaves a broken file): weights, optimizer,
    LR schedule, step, data position and RNG states
  * the data order of each epoch is a fixed permutation (seed + epoch), so a resumed
    run skips exactly the batches already trained on in the current epoch
  * running the same command again resumes automatically from ``latest.pt`` with the
    config saved in the run directory
  * ``touch <run_dir>/STOP`` -> saves and exits after the current step (recommended pause);
    Ctrl+C or ``kill`` (SIGTERM) do the same
  * every ``save_every_steps`` a weights-only ``checkpoints/step_XXXXXXX.pt`` is kept for
    evaluation (the RoboTwin script's checkpoint_epoch_N.pt equivalent)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler

from . import lila_api, models
from .config import save_config
from .data import build_dataset, collate
from .utils.repro import seed_everything, worker_init_fn, write_run_info

logger = logging.getLogger(__name__)

LATEST = "latest.pt"
STOP = "STOP"
CSV_FIELDS = ("step", "epoch", "loss", "loss_fm", "loss_ff", "lr", "sec_per_step")


def make_scheduler(optimizer, total_steps: int, warmup: int, lr: float, lr_min: float) -> LambdaLR:
    """Linear warm-up then cosine from lr to lr_min over total_steps optimizer steps."""
    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return (lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * p))) / lr
    return LambdaLR(optimizer, factor)


class ResumableRandomSampler(Sampler):
    """Each epoch: a fixed permutation from (seed + epoch); iteration can start at any index."""

    def __init__(self, n: int, seed: int = 0):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self) -> int:
        return max(0, self.n - self.start)


def atomic_save(obj, path: Path) -> None:
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def has_checkpoint(run_dir: Path) -> bool:
    return (Path(run_dir) / LATEST).is_file()


class Trainer:
    def __init__(self, cfg: DictConfig, run_dir: Path, device: str, dtype: torch.dtype,
                 init_from: str | None = None, fixed_batch: bool = False):
        self.cfg, self.run_dir, self.device, self.dtype = cfg, Path(run_dir), device, dtype
        self.fixed_batch = fixed_batch
        self.callbacks: list[Callable[["Trainer", dict], None]] = []   # extension hook
        self.resuming = has_checkpoint(self.run_dir) and not fixed_batch
        t = cfg.training
        if self.resuming and init_from:
            raise ValueError(f"{self.run_dir} already has {LATEST}; --init_from only starts NEW runs "
                             "(use another run name)")

        seed = int(cfg.experiment.seed)
        seed_everything(seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        stop = self.run_dir / STOP
        if stop.exists():
            stop.unlink()

        # ---- data (fixed per-epoch order so that a resume can skip exactly)
        self.dataset = build_dataset(cfg, "train")
        self.batch_size = int(t.batch_size)
        self.sampler = ResumableRandomSampler(len(self.dataset), seed=seed)
        self.loader = DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=self.sampler, drop_last=True,
            num_workers=int(cfg.system.num_workers), pin_memory=bool(cfg.system.pin_memory),
            collate_fn=collate, worker_init_fn=worker_init_fn,
            persistent_workers=int(cfg.system.num_workers) > 0,
        )
        self.batches_per_epoch = len(self.dataset) // self.batch_size
        if self.batches_per_epoch == 0:
            raise ValueError("Fewer samples than one batch; lower training.batch_size")

        # ---- self-contained run directory (written once; a resumed run keeps its originals)
        if not self.resuming:
            save_config(cfg, self.run_dir / "config.yaml")
            shutil.copy2(cfg.paths.norm_stats, self.run_dir / "norm_stats.json")
            if cfg.model.get("use_task_cond", False):
                vtt_dst = self.run_dir / "vtt"
                for task in sorted({ep.task for ep in self.dataset.episodes}):
                    (vtt_dst / task).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(cfg.dataset.task_cond_dir) / task / "task_cond.npy", vtt_dst / task)
            self.dataset.save_manifest(self.run_dir / "train_episodes.json")
            write_run_info(self.run_dir, {"dataset": self.dataset.describe(), "init_from": init_from,
                                          "resumes": []})

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
        self.data_epoch = 0          # epoch of the data order currently being consumed
        self.batch_in_epoch = 0      # batches of that epoch already consumed

        if init_from:
            ck = torch.load(init_from, map_location="cpu", weights_only=False)
            self.action_model.load_state_dict(ck["model_state_dict"], strict=True)
            logger.info(f"initialised weights from {init_from} (fresh optimizer/schedule)")
        if self.resuming:
            self._load_latest()

        self.csv_path = self.run_dir / "loss.csv"
        self._prepare_csv()
        self._stop_signal: str | None = None

    # ------------------------------------------------------------- resume
    def _load_latest(self) -> None:
        ck = torch.load(self.run_dir / LATEST, map_location="cpu", weights_only=False)
        self.action_model.load_state_dict(ck["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        for st in self.optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(self.device)
        self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        self.global_step = int(ck["global_step"])
        self.samples_seen = int(ck["samples_seen"])
        self.data_epoch = int(ck["data_epoch"])
        self.batch_in_epoch = int(ck["batch_in_epoch"])
        if "rng" in ck:
            _set_rng_state(ck["rng"])
        logger.info(f">>> resuming from {self.run_dir / LATEST}: step {self.global_step}/{self.total_steps}, "
                    f"data epoch {self.data_epoch + 1} after {self.batch_in_epoch}/{self.batches_per_epoch} "
                    f"batches, lr {self.optimizer.param_groups[0]['lr']:.2e}")
        info_path = self.run_dir / "run_info.json"
        try:
            info = json.loads(info_path.read_text())
            info.setdefault("resumes", []).append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                   "from_step": self.global_step})
            info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False))
        except (OSError, ValueError):
            pass

    def _prepare_csv(self) -> None:
        """New run: header. Resumed run: drop rows logged after the checkpoint (they are retrained)."""
        if self.resuming and self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["step"]) <= self.global_step]
            with open(self.csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
        elif not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_FIELDS)

    def save_latest(self, reason: str = "periodic") -> None:
        atomic_save({
            "model_state_dict": self.action_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step, "samples_seen": self.samples_seen,
            "data_epoch": self.data_epoch, "batch_in_epoch": self.batch_in_epoch,
            "total_steps": self.total_steps, "rng": _rng_state(),
            "finished": self.global_step >= self.total_steps,
        }, self.run_dir / LATEST)
        logger.info(f"[resume point] {LATEST} saved at step {self.global_step} ({reason})")

    def save_milestone(self) -> Path:
        """Weights-only checkpoint for evaluation (~0.4 GB)."""
        path = self.run_dir / "checkpoints" / f"step_{self.global_step:07d}.pt"
        atomic_save(lila_api.checkpoint_payload(
            self.action_model, global_step=self.global_step, samples_seen=self.samples_seen,
            epoch=self.epoch, config=OmegaConf.to_container(self.cfg, resolve=True)), path)
        (self.run_dir / "checkpoints" / "latest.txt").write_text(path.name)
        logger.info(f"saved {path}")
        return path

    # ------------------------------------------------------------------ data
    def _batches(self) -> Iterator[dict]:
        if self.fixed_batch:
            self.sampler.set_epoch(0, 0)
            batch = next(iter(self.loader))
            while True:
                yield batch
        while True:
            self.sampler.set_epoch(self.data_epoch, self.batch_in_epoch * self.batch_size)
            yield from self.loader
            self.data_epoch += 1
            self.batch_in_epoch = 0

    @property
    def epoch(self) -> float:
        return self.samples_seen / len(self.dataset)

    # ----------------------------------------------------------- stopping
    def _install_signal_handlers(self):
        def handler(signum, frame):
            if self._stop_signal is not None:          # second Ctrl+C: stop immediately
                raise KeyboardInterrupt
            self._stop_signal = signal.Signals(signum).name
            logger.info(f"received {self._stop_signal}: saving and stopping after the current step "
                        f"(press Ctrl+C again to abort without saving)")
        old = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
        return lambda: [signal.signal(s, h) for s, h in old.items()]

    def _should_stop(self) -> str | None:
        if self._stop_signal:
            return self._stop_signal
        if (self.run_dir / STOP).exists():
            (self.run_dir / STOP).unlink()
            return "STOP file"
        return None

    # ------------------------------------------------------------------ loop
    def train(self) -> str:
        """Returns 'finished' or 'stopped'."""
        t = self.cfg.training
        accum, clip = int(t.grad_accum_steps), float(t.grad_clip_norm)
        log_every, save_every = int(t.log_every_steps), int(t.save_every_steps)
        resume_every = int(t.get("resume_every_steps", 1000))
        lam = float(t.get("lambda_future_feat", 0.0))
        if self.global_step >= self.total_steps:
            logger.info(f"run already finished ({self.global_step}/{self.total_steps} steps)")
            return "finished"
        batches = self._batches()
        logger.info(f"training {self.total_steps} optimizer steps, effective batch {self.batch_size * accum}, "
                    f"{len(self.dataset)} samples, {self.batches_per_epoch} batches/epoch "
                    f"(~{self.total_steps * self.batch_size * accum / len(self.dataset):.1f} epochs); "
                    f"stop safely with: touch {self.run_dir / STOP}")
        restore = self._install_signal_handlers()
        tic = time.time()
        try:
            while self.global_step < self.total_steps:
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum = fm_sum = ff_sum = 0.0
                for _ in range(accum):
                    batch = next(batches)
                    self.batch_in_epoch += 1
                    with torch.autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu",
                                        dtype=self.dtype, enabled=self.dtype != torch.float32):
                        loss, info = self.model(batch)
                    (loss / accum).backward()
                    loss_sum += loss.item() / accum
                    fm_sum += float(info.get("loss_mse", 0.0)) / accum
                    ff_sum += float(info.get("loss_future_feat", 0.0)) / accum
                    self.samples_seen += self.batch_size
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
                        csv.writer(f).writerow([metrics[k] for k in CSV_FIELDS])
                    eta_h = dt * (self.total_steps - self.global_step) / 3600
                    logger.info(f"step {self.global_step}/{self.total_steps} ep {self.epoch:.2f} "
                                f"loss {loss_sum:.4f} (fm {fm_sum:.4f} + {lam}*ff {ff_sum:.4f}) "
                                f"lr {metrics['lr']:.2e} {dt:.2f}s/step ETA {eta_h:.1f}h")
                    for cb in self.callbacks:
                        cb(self, metrics)

                done = self.global_step == self.total_steps
                if self.global_step % save_every == 0 or done:
                    self.save_milestone()
                if not self.fixed_batch and (self.global_step % resume_every == 0 or done):
                    self.save_latest("finished" if done else "periodic")
                reason = self._should_stop()
                if reason and not done:
                    if not self.fixed_batch:
                        self.save_latest(reason)
                    logger.info(f"[paused] {reason}: stopped at step {self.global_step}. "
                                f"Run the same command again to resume.")
                    return "stopped"
        finally:
            restore()
        logger.info("Training complete.")
        return "finished"
