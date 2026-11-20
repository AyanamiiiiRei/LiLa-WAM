#!/usr/bin/env python
"""Can a rule on the robot state alone (computable online from the ee16 proprio) reproduce the
annotated sub-task boundaries of every task? CPU only, a few minutes, no model.

  python scripts/step44_state_segments.py --config configs/stage_memory/M_all.yaml

Question: if stages are segmented by state (instead of the timer), the same rule must give the
training boundaries and the online writes. Before training on it, check per task how well each
candidate rule matches the annotated boundaries (artifacts/subtasks) on the training demos.

Rules (ee16: left xyz 0:3, left gripper 7, right xyz 8:11, right gripper 15; gripper 1 = open):
  release       a gripper opens after having been closed for >= MIN_CLOSED steps
  release_home  as release, and the stage ends only when both arms are back within HOME_TOL of
                their start position (the phase-2 proprio rule of swap_blocks, generalised)
Per task: demos, annotated boundaries per demo (median), events per demo (median), recall =
boundaries with an event within +-TOL steps, precision = events within +-TOL of a boundary,
same count = demos where #events == #boundaries, median |offset| of the matched events.
Output: printed table + artifacts/stage_memory/state_segments_check.md
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import _common
import numpy as np

from rmbench_lila.episodes import EpisodeReader, select_episodes
from rmbench_lila.schema import STATE_GROUP, get_state_layout

MIN_CLOSED = 4
HOME_TOL = 0.02
TOL = 16
GRIP_OPEN = 0.5


def release_events(s):
    ev = []
    for g in (7, 15):
        closed = s[:, g] <= GRIP_OPEN
        run = 0
        for t in range(len(s)):
            if closed[t]:
                run += 1
            else:
                if run >= MIN_CLOSED:
                    ev.append(t)
                run = 0
    return sorted(ev)


def release_home_events(s):
    home_l, home_r = s[0, 0:3], s[0, 8:11]
    ev, armed, run = [], False, [0, 0]
    for t in range(len(s)):
        for i, g in enumerate((7, 15)):
            if s[t, g] <= GRIP_OPEN:
                run[i] += 1
                if run[i] >= MIN_CLOSED:
                    armed = True
            else:
                run[i] = 0
        if armed and s[t, 7] > GRIP_OPEN and s[t, 15] > GRIP_OPEN and \
                np.linalg.norm(s[t, 0:3] - home_l) < HOME_TOL and np.linalg.norm(s[t, 8:11] - home_r) < HOME_TOL:
            ev.append(t)
            armed = False
    return ev


def match(bounds, events):
    hit = sum(any(abs(e - b) <= TOL for e in events) for b in bounds)
    good = sum(any(abs(e - b) <= TOL for b in bounds) for e in events)
    offs = [min(abs(e - b) for e in events) for b in bounds if events and min(abs(e - b) for e in events) <= TOL]
    return hit, good, offs


def main():
    p = _common.parser(__doc__)
    args = p.parse_args()
    cfg = _common.load(args)
    from rmbench_lila.subtask import Library
    lib = Library(cfg.dataset.stage_memory.library_dir)
    layout = get_state_layout(cfg.rmbench.state_layout)
    refs = select_episodes(cfg, "train")
    rules = {"release": release_events, "release_home": release_home_events}
    acc = defaultdict(lambda: defaultdict(list))
    for r in refs:
        with EpisodeReader(r.path) as rd:
            s = rd.read_vector(layout, STATE_GROUP)
        rows = lib.segments[r.task][Path(r.path).stem]
        bounds = [row[2] for row in rows[:-1]]            # raw annotated ends, last sub-task excluded
        a = acc[r.task]
        a["nb"].append(len(bounds))
        for name, fn in rules.items():
            ev = fn(s)
            hit, good, offs = match(bounds, ev)
            a[name + "_ne"].append(len(ev))
            a[name + "_hit"].append(hit)
            a[name + "_good"].append(good)
            a[name + "_same"].append(len(ev) == len(bounds))
            a[name + "_off"] += offs
    lines = [f"tolerance +-{TOL} steps, gripper closed >= {MIN_CLOSED} steps, home tolerance {HOME_TOL} m", "",
             "| task | demos | boundaries / demo | rule | events / demo | recall | precision | same count | median offset |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task in sorted(acc):
        a = acc[task]
        nb = sum(a["nb"])
        for name in rules:
            ne = sum(a[name + "_ne"])
            rec = sum(a[name + "_hit"]) / max(nb, 1)
            prec = sum(a[name + "_good"]) / max(ne, 1)
            same = sum(a[name + "_same"])
            off = np.median(a[name + "_off"]) if a[name + "_off"] else float("nan")
            lines.append(f"| {task} | {len(a['nb'])} | {np.median(a['nb']):.0f} | {name} | {np.median(a[name + '_ne']):.0f} | "
                         f"{rec:.2f} | {prec:.2f} | {same}/{len(a['nb'])} | {off:.0f} |")
    text = "\n".join(lines)
    print(text)
    out = Path("artifacts/stage_memory/state_segments_check.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
