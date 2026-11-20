#!/usr/bin/env python
"""Check the per-task state-segmentation rules (rmbench_lila/state_segments.py) on ALL training
demos against the annotated sub-tasks. CPU only, a few minutes, no model.

  python scripts/step44_state_segments.py --config configs/stage_memory/M_all.yaml

A rule is usable for a task if, on (nearly) every demo,
  count     the number of events equals the number of stages the rule should end
            (default: one per sub-task, the last one optional because it ends the task;
             swap_T: "pick" and "move home" merge -> one fewer; press_button: one per press;
             observe_and_pickup: exactly one, "wait" ends when the arm starts to move)
  in place  event k lies inside the sub-task it should end, [start - 16, end + 16]
Reported per task: demos, sub-tasks per demo, events per demo, count ok, in place, median and
5-95 % offset (event - annotated end, env steps; negative = earlier), max events per demo
(must fit model.stage_memory.num_slots). press_button: presses per sub-task instead.
Output: printed + artifacts/stage_memory/state_segments_check.md
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import _common
import numpy as np

from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.schema import STATE_GROUP, get_state_layout
from rmbench_lila.state_segments import TASK_RULE, segment
from rmbench_lila.subtask import Library

SLACK = 16


def expected(task, rows):
    """-> list of (start, end) of the sub-tasks the events should end, in order; None = per press."""
    if task == "press_button":
        return None
    if task == "swap_T":
        rows = rows[1:]
    if task == "observe_and_pickup":
        return [(rows[0][2], rows[1][2])]          # the arm starts after "wait" ends (offset vs its end)
    return [(r[1], r[2]) for r in rows]


def main():
    p = _common.parser(__doc__)
    args = p.parse_args()
    cfg = _common.load(args)
    lib = Library(cfg.dataset.stage_memory.library_dir)
    layout = get_state_layout(cfg.rmbench.state_layout)
    n_slots = int(cfg.model.stage_memory.num_slots)
    acc = defaultdict(lambda: defaultdict(list))
    for r in select_episodes(cfg, "train"):
        with EpisodeReader(r.path) as rd:
            s = rd.read_vector(layout, STATE_GROUP)
        rows = lib.segments[r.task][Path(r.path).stem]
        ev = segment(s, r.task)
        a = acc[r.task]
        a["nsub"].append(len(rows))
        a["nev"].append(len(ev))
        exp = expected(r.task, rows)
        if exp is None:                                  # press_button: presses per sub-task
            a["per_sub"].append(tuple(sum(st <= e <= en + SLACK for e in ev) for _, st, en in rows))
            a["outside"].append(sum(not any(st <= e <= en + SLACK for _, st, en in rows[:-1]) for e in ev))
            continue
        ok_count = len(ev) in (len(exp), len(exp) - 1) if r.task != "observe_and_pickup" else len(ev) == 1
        a["count_ok"].append(ok_count)
        inplace = all(k < len(exp) and exp[k][0] - SLACK <= e <= exp[k][1] + SLACK for k, e in enumerate(ev))
        a["in_place"].append(inplace and ok_count)
        ref = [rows[0][2]] if r.task == "observe_and_pickup" else [en for _, en in exp]
        a["off"] += [e - ref[k] for k, e in enumerate(ev) if k < len(ref)]
    lines = [f"rules: {TASK_RULE}", f"slack +-{SLACK} steps, num_slots {n_slots}", "",
             "| task | rule | demos | sub-tasks / demo | events / demo | count ok | in place | offset median [5%, 95%] | max events |",
             "|---|---|---|---|---|---|---|---|---|"]
    ok_all = True
    for task in sorted(acc):
        a = acc[task]
        n = len(a["nsub"])
        sub = dict(sorted(Counter(a["nsub"]).items()))
        evd = dict(sorted(Counter(a["nev"]).items()))
        mx = max(a["nev"])
        if task == "press_button":
            per = Counter(a["per_sub"]).most_common(6)
            lines.append(f"| {task} | {TASK_RULE[task]} | {n} | {sub} | {evd} | – | presses outside left/middle: "
                         f"{sum(a['outside'])} | presses per (left, middle, confirm): {per} | {mx} |")
            ok = sum(a["outside"]) == 0 and mx <= n_slots
        else:
            off = np.array(a["off"]) if a["off"] else np.zeros(1)
            lines.append(f"| {task} | {TASK_RULE[task]} | {n} | {sub} | {evd} | {sum(a['count_ok'])}/{n} | "
                         f"{sum(a['in_place'])}/{n} | {np.median(off):+.0f} [{np.percentile(off, 5):+.0f}, "
                         f"{np.percentile(off, 95):+.0f}] | {mx} |")
            ok = sum(a["in_place"]) >= 0.9 * n and mx <= n_slots
        ok_all &= ok
        if not ok:
            lines.append(f"|   ^ {task}: NOT usable as is |||||||||")
    lines += ["", "ALL USABLE" if ok_all else "SOME TASKS NOT USABLE (see ^ lines)"]
    text = "\n".join(lines)
    print(text)
    out = Path("artifacts/stage_memory/state_segments_check.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
