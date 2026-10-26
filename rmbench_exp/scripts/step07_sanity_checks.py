#!/usr/bin/env python
"""Plan step 7: checks to run BEFORE the real training. Do not skip them.

Sub-commands (each writes into artifacts/sanity/<name>/):
  color    dataloader image (after normalise -> denormalise) next to the mp4 frame  [CPU]
  actions  14-d action and 16-d state curves of a few episodes; flags jumps        [CPU]
  shapes   one forward pass; prints every token group + the DiT input sequence     [GPU, DINOv3]
  overfit  train repeatedly on ONE batch; loss must fall close to zero             [GPU, DINOv3]

Usage:
  python scripts/step07_sanity_checks.py color   --config configs/exp/baseline.yaml
  python scripts/step07_sanity_checks.py overfit --config configs/exp/debug_overfit.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import _common
import cv2
import numpy as np

from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.image import from_model_input
from rmbench_lila.schema import STATE_GROUP, get_action_layout, get_state_layout

OUT = _common.SUBPROJECT_ROOT / "artifacts" / "sanity"


def cmd_color(cfg, args):
    from rmbench_lila.data import build_dataset
    out = OUT / "color"
    out.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(cfg, "train")
    rng = np.random.default_rng(args.seed)
    for i in rng.choice(len(ds), size=min(args.n, len(ds)), replace=False):
        s = ds[int(i)]
        img = from_model_input(s["pixel_values"].numpy())
        ep = ds.episodes[ds.index[int(i)][0]]
        cap = cv2.VideoCapture(str(ep.video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s["anchor"])
        ok, bgr = cap.read()
        cap.release()
        panels = [img]
        if ok:
            vid = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (img.shape[1], img.shape[0]))
            panels.append(vid)
        name = f"{s['task_name']}_ep{s['episode_id']}_t{s['anchor']}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(np.concatenate(panels, 1), cv2.COLOR_RGB2BGR))
        print(f"{name}: left = dataloader, right = {'mp4' if ok else '(no video found)'}")
    print(f"\nOpen {out} and check that both halves have identical colours (red stays red).")


def cmd_actions(cfg, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = OUT / "actions"
    out.mkdir(parents=True, exist_ok=True)
    a_lay, s_lay = get_action_layout(cfg.rmbench.action_layout), get_state_layout(cfg.rmbench.state_layout)
    refs = select_episodes(cfg, "train")
    by_task = {}
    for r in refs:
        by_task.setdefault(r.task, r)
    report = {}
    for task, ref in sorted(by_task.items()):
        with EpisodeReader(ref.path) as rd:
            act = rd.read_action(a_lay, cfg.rmbench.action_alignment)
            st = rd.read_vector(s_lay, STATE_GROUP)
        step = np.abs(np.diff(act, axis=0))
        jumps = {int(d): float(step[:, d].max()) for d in range(act.shape[1]) if step[:, d].max() > args.jump}
        report[task] = {"len": int(act.shape[0]), "max_step_per_dim": step.max(0).round(4).tolist(), "jumps": jumps}
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        names = [f"{f.h5_key}[{k}]" for f, sl in a_lay.slices() for k in range(sl.stop - sl.start)]
        for d in range(act.shape[1]):
            axes[0].plot(act[:, d], label=names[d], lw=1)
        axes[0].set_title(f"{task} ep{ref.episode_id}: action ({a_lay.name}, alignment={cfg.rmbench.action_alignment})")
        axes[0].legend(fontsize=6, ncol=7)
        for d in range(st.shape[1]):
            axes[1].plot(st[:, d], lw=1)
        axes[1].set_title(f"state ({s_lay.name})")
        fig.tight_layout()
        fig.savefig(out / f"{task}.png", dpi=110)
        plt.close(fig)
        print(f"{task}: len {act.shape[0]}, jumps > {args.jump}: {jumps or 'none'}")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"\nPlots in {out}. Check: left arm = dims 0-5, left gripper = 6, right arm = 7-12, "
          f"right gripper = 13; curves smooth; grippers between 0 and 1.")


def _build(cfg, args, training):
    from rmbench_lila import models
    device, dtype = _common.pick_device_dtype(args.device)
    built = models.build(cfg, cfg.paths.norm_stats, device, dtype, training=training)
    built.action_model.to(device, dtype)
    return built, device, dtype


def cmd_shapes(cfg, args):
    import torch
    from rmbench_lila import lila_api
    from rmbench_lila.data import build_dataset, collate
    ds = build_dataset(cfg, "train")
    batch = collate([ds[i] for i in range(2)])
    built, device, dtype = _build(cfg, args, training=True)
    for k, v in batch.items():
        print(f"batch[{k!r}]: {tuple(v.shape) if hasattr(v, 'shape') else v}")
    seen = {}
    h = lila_api.dit_blocks(built.action_model)[0].register_forward_pre_hook(
        lambda m, inp: seen.setdefault("x", tuple(inp[0].shape)))
    with torch.autocast("cuda", dtype=dtype, enabled=device.startswith("cuda")):
        loss, info = built.wrapper(batch)
    h.remove()
    c = cfg.common
    q = cfg.model.action_expert.vlm_adapter_num_queries
    expect = c.action_chunk_size + c.proprio_len + cfg.model.num_registers + int(cfg.model.use_task_cond) + q
    print(f"\nDiT input sequence: {seen.get('x')}  expected (B, {expect}, {cfg.model.action_expert.hidden_size}) = "
          f"{c.action_chunk_size} action + {c.proprio_len} state + {cfg.model.num_registers} registers + "
          f"{int(cfg.model.use_task_cond)} VTT + {q} visual")
    print(f"loss {loss.item():.4f}  fm {info['loss_mse']:.4f}  ff {info['loss_future_feat']:.4f}")
    assert seen.get("x", (0, -1))[1] == expect, "unexpected token count"

    # ---- peak GPU memory of one training step at the configured batch size
    if not device.startswith("cuda"):
        return
    bs = int(args.mem_batch or cfg.training.batch_size)
    rng = np.random.default_rng(args.seed)
    big = collate([ds[int(i)] for i in rng.choice(len(ds), size=bs, replace=False)])
    del loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    built.action_model.train()
    try:
        with torch.autocast("cuda", dtype=dtype):
            loss, _ = built.wrapper(big)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        # AdamW adds two fp-state tensors per trainable parameter on the first optimizer step
        n_train = sum(p.numel() for p in built.action_model.parameters() if p.requires_grad)
        adam = 2 * n_train * torch.finfo(dtype).bits / 8 / 2**30
        print(f"\nbatch_size={bs}: peak {peak:.1f} GiB (+ ~{adam:.1f} GiB AdamW state) of {total:.1f} GiB")
        print("OK" if peak + adam < 0.9 * total else
              "TOO TIGHT: lower training.batch_size and raise grad_accum_steps (keep the product = 128)")
    except torch.cuda.OutOfMemoryError:
        print(f"\nbatch_size={bs}: OUT OF MEMORY. Retry with --mem_batch {bs // 2}, then set "
              f"training.batch_size={bs // 2} training.grad_accum_steps={2 * int(cfg.training.grad_accum_steps)}")


def cmd_overfit(cfg, args):
    from rmbench_lila.trainer import Trainer
    from rmbench_lila.utils.repro import timestamp
    device, dtype = _common.pick_device_dtype(args.device)
    run = OUT / "overfit" / timestamp()
    tr = Trainer(cfg, run, device, dtype, fixed_batch=True)
    tr.train()
    import csv
    rows = list(csv.DictReader(open(run / "loss.csv")))
    # noise and flow time are resampled every step, so the loss is noisy: compare averages
    first = float(np.mean([float(r["loss_fm"]) for r in rows[:3]]))
    last = float(np.mean([float(r["loss_fm"]) for r in rows[-5:]]))
    print(f"\nflow-matching loss on one batch: {first:.4f} -> {last:.4f} (x{first / max(last, 1e-9):.1f} lower)")
    print("OK: the model can fit the data." if last < 0.2 * first else
          "WARNING: loss fell less than 5x. Rerun with training.total_steps=1000; if it still does not "
          "fall, check data/normalisation before real training.")


def main():
    p = _common.parser(__doc__, config=False)
    # declared before the free-form overrides so that "color key=value" parses correctly
    p.add_argument("check", choices=["color", "actions", "shapes", "overfit"])
    p.add_argument("--config", default=str(_common.DEFAULT_CONFIG))
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. training.batch_size=8")
    p.add_argument("--n", type=int, default=6, help="color: number of samples")
    p.add_argument("--jump", type=float, default=0.2, help="actions: flag per-step changes above this (rad)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mem_batch", type=int, default=None, help="shapes: batch size for the memory test "
                   "(default training.batch_size)")
    args = p.parse_args()
    cfg = _common.load(args)
    {"color": cmd_color, "actions": cmd_actions, "shapes": cmd_shapes, "overfit": cmd_overfit}[args.check](cfg, args)


if __name__ == "__main__":
    main()
