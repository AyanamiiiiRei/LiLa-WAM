#!/usr/bin/env python
"""Sub-task design, step 1: split every demo into sub-tasks and build the sub-task VTT library.

  python scripts/step36_subtask_library.py            # all 10 tasks (needs step 21 CLS caches)

Sub-tasks come from RMBench's language annotations: consecutive segments with the same
description are merged (one sub-task = one merged segment). Exception: swap_blocks, whose
description changes in the middle of a move; it uses the event boundaries of the stage-memory
experiment (artifacts/stage_memory/swap_blocks/boundaries_gt.json) and the names move 1-3, press.

A sub-task TYPE = (task, description). Its VTT = mean over all its segments of
CLS(last frame) - CLS(first frame) (last-layer DINOv3 CLS, the same definition as the task VTT,
applied to one segment). The description already carries some episode-specific content
("move the block back to the FRONT mat"), so choosing the right type is part of the memory
problem (the chooser's job); what the text does not say (which object was shown, which mat
was the original one) has to come from the memory entry the chooser hands to the executor.

Outputs (artifacts/subtasks/):
  library.npz       names (K,), task (K,), vtt (K, 1024), count (K,)
  segments.json     {task: {episode: [[type_id, start, end], ...]}}  (raw frames, end inclusive)
  report.md         per task: segments per demo, types, VTT norm and consistency, short segments
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import _common
import numpy as np

from rmbench_lila.stages import merge_annotation, raw_ends

TASKS = ["battery_try", "blocks_ranking_try", "cover_blocks", "observe_and_pickup", "place_block_mat",
         "press_button", "put_back_block", "rearrange_blocks", "swap_T", "swap_blocks"]
# what the chosen description does NOT tell the executor (must come from a memory entry)
NOT_IN_TEXT = {
    "observe_and_pickup": "which object was shown at the start ('pick up the target object')",
    "place_block_mat": "which mat the block came from ('put it back to the original mat')",
    "cover_blocks": "which cover hides which colour ('open ... in the order of red, green, blue')",
}


def main():
    p = _common.parser(__doc__)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    p.add_argument("--out", default="artifacts/subtasks")
    args = p.parse_args()
    cfg = _common.load(args)
    root = Path(cfg.rmbench.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vtts: dict = defaultdict(list)            # (task, name) -> list of segment VTTs
    segs: dict = {}
    per_task: dict = {}
    for task in args.tasks:
        cls_dir = Path(f"artifacts/stage_memory/{task}/cls")
        files = sorted(cls_dir.glob("episode_*.npy"))
        if not files:
            raise SystemExit(f"{task}: no CLS cache in {cls_dir} (python scripts/step21_cache_cls.py --task {task})")
        ann = json.loads((root / task / cfg.rmbench.embodiment / "language_annotation.json").read_text())
        bounds = None
        if task == "swap_blocks":
            bounds = json.loads(Path("artifacts/stage_memory/swap_blocks/boundaries_gt.json").read_text())
        segs[task], n_per, short = {}, [], 0
        for f in files:
            cls = np.load(f).astype(np.float32)
            n = len(cls)
            idx = int(f.stem.split("_")[1])
            if bounds is not None:
                stages = bounds[f.stem + ".hdf5"]
                names = ["move 1", "move 2", "move 3", "press"]
                pieces = [(names[k], s, min(e, n) - 1) for k, (s, e) in enumerate(stages)]
                pieces[-1] = (pieces[-1][0], pieces[-1][1], n - 1)
            else:
                merged = merge_annotation(ann[f"episode_{idx}"])
                ends = [min(e, n - 1) for e in raw_ends(merged)]
                starts = [0] + [e + 1 for e in ends[:-1]]
                pieces = [(d, s, e) for (d, _), s, e in zip(merged, starts, ends) if e > s]
            row = []
            for name, s, e in pieces:
                vtts[(task, name)].append(cls[e] - cls[s])
                row.append([name, int(s), int(e)])
                short += (e - s + 1) < 32
            segs[task][f.stem] = row
            n_per.append(len(row))
        per_task[task] = (len(files), n_per, short)

    keys = sorted(vtts)
    names = np.array([k[1] for k in keys])
    tasks = np.array([k[0] for k in keys])
    mean = np.stack([np.mean(vtts[k], axis=0) for k in keys]).astype(np.float32)
    count = np.array([len(vtts[k]) for k in keys])
    np.savez(out / "library.npz", names=names, task=tasks, vtt=mean, count=count)
    tid = {k: i for i, k in enumerate(keys)}
    (out / "segments.json").write_text(json.dumps(
        {t: {ep: [[tid[(t, n)], s, e] for n, s, e in row] for ep, row in d.items()} for t, d in segs.items()}))

    def consistency(k):
        v = np.stack(vtts[k])
        m = mean[tid[k]]
        return float(np.mean(v @ m / (np.linalg.norm(v, axis=1) * np.linalg.norm(m) + 1e-8)))

    lines = [f"# Sub-task library ({len(keys)} types, {len(args.tasks)} tasks)", "",
             "| task | demos | sub-tasks per demo | types | segments < 32 frames | not in the text (needs a memory entry) |",
             "|---|---|---|---|---|---|"]
    for t in args.tasks:
        nd, n_per, short = per_task[t]
        dist = ", ".join(f"{k}: {n_per.count(k)}" for k in sorted(set(n_per)))
        lines.append(f"| {t} | {nd} | {dist} | {sum(tasks == t)} | {short} | {NOT_IN_TEXT.get(t, '-')} |")
    lines += ["", "| task | sub-task type | segments | VTT norm | consistency (mean cos to the type mean) |",
              "|---|---|---|---|---|"]
    for k in keys:
        lines.append(f"| {k[0]} | {k[1][:80]} | {count[tid[k]]} | {np.linalg.norm(mean[tid[k]]):.2f} | {consistency(k):.2f} |")
    # can the type be told apart from its VTT? nearest-mean accuracy, leave-one-segment-out not needed for a sketch
    allv = np.concatenate([np.stack(vtts[k]) for k in keys])
    alll = np.concatenate([[tid[k]] * len(vtts[k]) for k in keys])
    mn = mean / np.linalg.norm(mean, axis=1, keepdims=True)
    pred = (allv / (np.linalg.norm(allv, axis=1, keepdims=True) + 1e-8) @ mn.T).argmax(1)
    lines += ["", f"segment VTT -> nearest type mean (cosine), all segments: {(pred == alll).mean():.2f} "
              f"(within the same task only: {np.mean([tasks[a] == tasks[b] for a, b in zip(pred, alll)]):.2f} "
              "of the predictions land in the right task)"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
