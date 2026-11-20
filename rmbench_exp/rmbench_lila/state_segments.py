"""Stage segmentation from the robot state alone (docs/STAGE_MEMORY.md, "状态分段").

A stage ends at a STATE EVENT computed from the ee16 proprio only (left xyz 0:3, left gripper 7,
right xyz 8:15 -> xyz 8:11, right gripper 15; gripper 1 = open). The SAME class gives the
training boundaries (run over a demo, `segment`) and the online writes during evaluation
(`update` once per env step), so training and evaluation always agree. The annotated sub-tasks
are only used to check the rules (scripts/step44_state_segments.py).

Primitive events
  release     a gripper opens after being closed for >= MIN_CLOSED steps
  press       an arm's z dips >= PRESS_DROP and rises again, the gripper closed for >= PRESS_MIN_CLOSED
              steps at the bottom and still closed when rising (grasp: closes only at the bottom;
              placement: opens at the bottom -> both excluded)
  home_all    both arms are within HOME_TOL of their start position, at least one of them left
              it since the last event, and a gripper was closed >= MIN_CLOSED steps meanwhile
  press_home  an arm returns to its start position after closing its gripper without releasing
              it (the arm that pressed a button)
  move_start  first time an arm is >= MOVE_TOL from its start position

Per-task rule (TASK_RULE): which primitive events end a stage.
"""

from __future__ import annotations

import numpy as np

MIN_CLOSED = 4
GRIP_OPEN = 0.5
HOME_TOL = 0.02
MOVE_TOL = 0.005
PRESS_DROP = 0.015
PRESS_MIN_CLOSED = 8
ARMS = ((slice(0, 3), 7), (slice(8, 11), 15))

TASK_RULE = {
    "swap_blocks": "home_all",           # each move ends when both arms are back
    "place_block_mat": "home_all",
    "battery_try": "home_all",           # dual-arm placement ends when BOTH arms are back
    "swap_T": "home_all",                # pick + move home merge into one stage
    "rearrange_blocks": "release_or_press_home",   # the left arm leaves for the button before the right is back
    "put_back_block": "release_or_press_home",
    "cover_blocks": "release",           # one grasp-release per cover / uncover
    "blocks_ranking_try": "press_or_3release",   # press ends "press"; 3rd release after it ends "swap"
    "press_button": "press",             # one stage per press (the count enters the memory)
    "observe_and_pickup": "move_start",  # end of "wait": the arm starts to move
}


class _Arm:
    def __init__(self, xyz):
        self.start = np.asarray(xyz, np.float64).copy()
        self.closed_run = 0
        self.closed_since_open = False     # closed >= MIN_CLOSED since the gripper last opened
        self.away = False
        self.closed_while_away = False
        self.moved = False
        # press detector
        self.dir = "down"
        self.peak = float(xyz[2])
        self.trough = float(xyz[2])
        self.trough_closed_run = 0

    def step(self, xyz, closed: bool) -> dict:
        ev = {"release": False, "press": False, "home": False, "press_home": False, "move_start": False}
        # gripper
        if closed:
            self.closed_run += 1
            if self.closed_run >= MIN_CLOSED:
                self.closed_since_open = True
        else:
            if self.closed_since_open:
                ev["release"] = True
            self.closed_run = 0
            self.closed_since_open = False
        # home / away
        d = float(np.linalg.norm(np.asarray(xyz) - self.start))
        if not self.moved and d >= MOVE_TOL:
            self.moved = True
            ev["move_start"] = True
        if d >= HOME_TOL:
            self.away = True
            if self.closed_run >= MIN_CLOSED:
                self.closed_while_away = True
        elif self.away:
            ev["home"] = True
            if self.closed_while_away and self.closed_run >= MIN_CLOSED:
                ev["press_home"] = True             # came back still closed: it pressed, not placed
            self.away = False
            self.closed_while_away = False
        # press: a dip whose bottom is reached with the gripper already closed for
        # >= PRESS_MIN_CLOSED steps and that rises again still closed (a grasp closes only at
        # the bottom; a placement opens at the bottom before rising)
        z = float(xyz[2])
        if self.dir == "down":
            if z < self.trough:
                self.trough, self.trough_closed_run = z, self.closed_run
            if z > self.trough + PRESS_DROP:
                if self.peak - self.trough >= PRESS_DROP and self.trough_closed_run >= PRESS_MIN_CLOSED and closed:
                    ev["press"] = True
                self.dir, self.peak = "up", z
        else:
            if z > self.peak:
                self.peak = z
            if z < self.peak - PRESS_DROP:
                self.dir, self.trough, self.trough_closed_run = "down", z, self.closed_run
        return ev


class StateSegmenter:
    """update(ee16 state) once per env step -> number of stages ended so far."""

    def __init__(self, task: str, max_stages: int | None = None):
        if task not in TASK_RULE:
            raise KeyError(f"no state-segmentation rule for task {task!r}")
        self.task, self.rule, self.max_stages = task, TASK_RULE[task], max_stages
        self.reset()

    def reset(self) -> None:
        self.arms = None
        self.t = -1
        self.done = 0
        self.events: list[int] = []
        self._gripped = False           # for home_all: a gripper closed since the last event
        self._releases_since_press = 0

    def update(self, state) -> int:
        s = np.asarray(state, np.float64).reshape(-1)
        self.t += 1
        if self.arms is None:
            self.arms = [_Arm(s[x]) for x, _ in ARMS]
        evs = [a.step(s[x], s[g] <= GRIP_OPEN) for a, (x, g) in zip(self.arms, ARMS)]
        if any(a.closed_run >= MIN_CLOSED for a in self.arms):
            self._gripped = True
        n = 0
        r = self.rule
        if r == "release":
            n = sum(e["release"] for e in evs)
        elif r == "release_or_press_home":
            n = sum(e["release"] or e["press_home"] for e in evs)
        elif r == "press":
            n = sum(e["press"] for e in evs)
        elif r == "press_or_3release":
            for e in evs:
                if e["press"]:
                    n += 1
                    self._releases_since_press = 0
                elif e["release"]:
                    self._releases_since_press += 1
                    if self._releases_since_press == 3:
                        n += 1
        elif r == "home_all":
            home = all(np.linalg.norm(s[x] - a.start) < HOME_TOL for a, (x, _) in zip(self.arms, ARMS))
            if home and self._gripped and any(e["home"] for e in evs):
                n = 1
                self._gripped = False
        elif r == "move_start":
            n = int(self.done == 0 and any(e["move_start"] for e in evs))
        for _ in range(n):
            if self.max_stages is None or self.done < self.max_stages:
                self.done += 1
                self.events.append(self.t)
        return self.done


def segment(states, task: str, max_stages: int | None = None) -> list[int]:
    """Event times (env steps) of a whole state sequence (T, 16)."""
    seg = StateSegmenter(task, max_stages)
    for s in np.asarray(states):
        seg.update(s)
    return list(seg.events)
