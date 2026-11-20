#!/usr/bin/env python
"""Stage memory on all tasks (configs/stage_memory/M_all.yaml): per-task time triggers and a
data check, before training. CPU only, a few minutes.

  python scripts/step42_time_triggers.py --config configs/stage_memory/M_all.yaml

1. Time trigger per task -> artifacts/stage_memory/<task>/time_trigger_10task.json
   ("ends_env_steps"): end of stage k = mean grid end of stage k over the TRAINING demos, rounded
   up to the 16-step grid, made strictly increasing. Stage k is kept only if at least half of
   the training demos have it (battery_try, blocks_ranking_try: the number of sub-tasks varies;
   press_button: the durations vary with the press counts -> a timer writes wrongly there).
   The phase-2 file time_trigger.json of swap_blocks is not touched.
2. Data check: for every task, samples of the training split through the stage_memory feature:
   number of written slots never exceeds the number of finished sub-tasks; prints the slot
   distribution. Exits non-zero on any problem.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import _common
import numpy as np

from rmbench_lila.data.dataset import RMBenchDataset
from rmbench_lila.data.stage_features import StageMemoryFeature
from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.stages import GRID
from rmbench_lila.subtask import Library

OUT_NAME = "time_trigger_10task.json"


def main():
    p = _common.parser(__doc__)
    p.add_argument("--samples_per_task", type=int, default=40)
    args = p.parse_args()
    cfg = _common.load(args)
    lib = Library(cfg.dataset.stage_memory.library_dir)
    train = select_episodes(cfg, "train")
    by_task = defaultdict(list)
    for r in train:
        by_task[r.task].append(r)
    n_slots = int(cfg.model.stage_memory.num_slots)
    ok = True
    print("task | training demos | sub-tasks per demo | time-trigger ends (env steps)")
    for task in sorted(by_task):
        ends_all = []
        for r in by_task[task]:
            _, ends = lib.episode(task, Path(r.path).stem, 10 ** 9)     # no truncation needed here
            ends_all.append(ends)
        counts = Counter(len(e) + 1 for e in ends_all)
        out, prev = [], 0
        for k in range(max(len(e) for e in ends_all)):
            have = [e[k] for e in ends_all if len(e) > k]
            if len(have) < len(ends_all) / 2:
                break
            g = max(int(math.ceil(float(np.mean(have)) / GRID) * GRID), prev + GRID)
            out.append(g)
            prev = g
        if len(out) > n_slots:
            print(f"ERROR {task}: {len(out)} stages > num_slots {n_slots}")
            ok = False
        art = Path("artifacts/stage_memory") / task
        art.mkdir(parents=True, exist_ok=True)
        (art / OUT_NAME).write_text(json.dumps({
            "ends_env_steps": out,
            "source": "mean grid ends of the training demos (sub-task library), stages present in >= half of them",
            "demos": len(ends_all), "subtasks_per_demo": dict(sorted(counts.items()))}))
        print(f"{task} | {len(ends_all)} | {dict(sorted(counts.items()))} | {out}")

    print("\ndata check (stage_memory feature, training split)")
    ds = RMBenchDataset(cfg, "train", episodes=train)
    feat = StageMemoryFeature(cfg, train=False)
    n_ends = defaultdict(list)
    print(f"segmentation: {feat.segmentation}")
    rng = random.Random(0)
    per_task = defaultdict(list)
    for i, (e, _) in enumerate(ds.index):
        per_task[ds.episodes[e].task].append(i)
    for task in sorted(per_task):
        idx = rng.sample(per_task[task], min(args.samples_per_task, len(per_task[task])))
        written = Counter()
        for i in idx:
            e, t = ds.index[i]
            ref = ds.episodes[e]
            item = ds[i]
            w = item["stage_mem_written"].numpy()
            with EpisodeReader(ref.path) as rd:      # boundaries exactly as the feature makes them
                stages = feat._stages(ref, int(ds.lengths[e]), rd)        # (annotation or state)
            ends = [b for _, b in stages[:-1]]
            finished = sum(t >= g for g in ends[:n_slots])
            n_ends[task].append(len(ends))
            # jitter moves an end by +-1 chunk: allow a difference of one slot
            if w.sum() > finished + 1 or w.sum() < finished - 1 or (w.sum() and not w[: w.sum()].all()):
                print(f"ERROR {task} {Path(ref.path).name} t={t}: written {w.astype(int).tolist()}, finished {finished}")
                ok = False
            if item["stage_mem_cls"].shape != (n_slots, 2, item["stage_mem_cls"].shape[-1]):
                print(f"ERROR {task}: stage_mem_cls shape {tuple(item['stage_mem_cls'].shape)}")
                ok = False
            written[int(w.sum())] += 1
        print(f"  {task}: written slots over {len(idx)} samples {dict(sorted(written.items()))}; "
              f"boundaries per sampled demo {dict(sorted(Counter(n_ends[task]).items()))}")
    print("OK" if ok else "PROBLEMS FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
