#!/usr/bin/env python
"""Sub-task design, plan 2: train the Sub Task Chooser on a FROZEN executor; or (any sub-task
run, --eval_only) report the chooser on the held-out demos.

  python scripts/step38_train_chooser.py --run runs/sub_P2-seed0 [--steps 3000]
  python scripts/step38_train_chooser.py --run runs/sub_P1-seed0 --eval_only

Data: every chunk-grid frame (t = 0, 16, ...) of the training demos (all tasks). For each,
the executor runs once at flow time 0 with the PRE-decision sub-task state (no gradient) and
the chooser learns: has the current sub-task ended at t (BCE, positives weighted 4x), and if
so, which type comes next (cross-entropy, among the task's types). Only the chooser's
parameters change. Training writes <run>/checkpoints/step_<N>.pt with the trained chooser
(the executor-only file is kept as step_<N>_executor.pt) and <run>/chooser_report.md.
Report (held-out demos, chunk grid): end precision / recall / F1, next-type accuracy at the true
ends, and end timing as the chooser would run along each demo (its own switches).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import _common
import numpy as np
import torch
from omegaconf import OmegaConf

from rmbench_lila import models
from rmbench_lila.data.dataset import RMBenchDataset, collate
from rmbench_lila.episodes import select_episodes
from rmbench_lila.policy import resolve_checkpoint
from rmbench_lila.subtask import chooser_loss, chooser_pass, executor_seq, run_chooser


def build(run, device, dtype):
    cfg = OmegaConf.load(run / "config.yaml")
    built = models.build(cfg, str(run / "norm_stats.json"), device, dtype, training=True)
    ck_path = resolve_checkpoint(run, "latest")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    built.action_model.load_state_dict(ck["model_state_dict"], strict=True)
    built.wrapper.to(device, dtype)
    return cfg, built, ck, ck_path


def grid_loader(cfg, split, batch, shuffle, workers):
    OmegaConf.update(cfg, "dataset.anchor_stride", 16, merge=True)
    ds = RMBenchDataset(cfg, split, episodes=select_episodes(cfg, split))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle, collate_fn=collate,
                                       num_workers=workers, drop_last=shuffle)


@torch.no_grad()
def evaluate(cfg, built, device, workers):
    am = built.action_model
    am.eval()
    tp = fp = fn = ok = n_end = 0
    for batch in grid_loader(cfg, "holdout", 64, False, workers):
        sel = batch["sub_end"] >= 0
        if not sel.any():
            continue
        with torch.autocast("cuda", dtype=built.wrapper.dtype):
            end_logit, nxt = chooser_pass(built.wrapper, am, batch, sel, grad=False)
        _, st = chooser_loss(end_logit, nxt, batch["sub_end"][sel].to(device), batch["sub_next"][sel].to(device))
        tp += st["tp"]; fp += st["fp"]; fn += st["fn"]; ok += st["next_ok"]; n_end += st["n_end"]
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-8), "next_acc": ok / max(n_end, 1),
            "n_end": n_end}


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int, default=6000)   # x 64 ~ 20 epochs of the ~19 k grid frames
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run = Path(args.run)
    device, dtype = _common.pick_device_dtype(args.device)
    cfg, built, ck, ck_path = build(run, device, dtype)
    workers = int(cfg.system.num_workers)
    am = built.action_model
    lines = [f"# Sub Task Chooser ({run.name}, executor step {ck.get('global_step')})", ""]
    if not args.eval_only:
        for prm in am.parameters():
            prm.requires_grad_(False)
        for prm in am.chooser.parameters():
            prm.requires_grad_(True)
        opt = torch.optim.AdamW(am.chooser.parameters(), lr=args.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        step = 0
        while step < args.steps:
            for batch in grid_loader(cfg, "train", args.batch, True, workers):
                am.eval()
                am.chooser.train()
                sel = batch["sub_end"] >= 0
                with torch.autocast("cuda", dtype=dtype):
                    seq = executor_seq(built.wrapper, am, batch, sel, grad=False)   # frozen executor
                    end_logit, nxt = run_chooser(am, batch, sel, seq)
                    loss, st = chooser_loss(end_logit, nxt, batch["sub_end"][sel].to(device),
                                            batch["sub_next"][sel].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                step += 1
                if step % 100 == 0 or step == 1:
                    print(f"chooser step {step}/{args.steps} loss {float(loss):.4f} ends {st['n_end']} "
                          f"tp {st['tp']} fp {st['fp']} fn {st['fn']} next {st['next_ok']}", flush=True)
                if step >= args.steps:
                    break
        exe = ck_path.with_name(ck_path.stem + "_executor.pt")
        if not exe.exists():
            shutil.copy2(ck_path, exe)
        ck["model_state_dict"] = {k: v.detach().cpu() for k, v in am.state_dict().items()}
        ck["chooser_steps"] = args.steps
        torch.save(ck, ck_path.with_suffix(".tmp"))
        ck_path.with_suffix(".tmp").replace(ck_path)
        lines.append(f"chooser trained {args.steps} steps on the frozen executor; saved into {ck_path.name} "
                     f"(executor-only copy: {exe.name})")
    m = evaluate(cfg, built, device, workers)
    lines += ["", "held-out demos, chunk grid:", "",
              "| end precision | end recall | end F1 | next-type accuracy at true ends | true ends |",
              "|---|---|---|---|---|",
              f"| {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {m['next_acc']:.2f} | {m['n_end']} |"]
    (run / "chooser_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
