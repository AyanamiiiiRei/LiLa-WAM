"""Evaluation-time stage oracle (stage-memory experiment, phase 1).

Runs in the RMBench SIMULATOR process (it reads privileged simulator state) and only
READS the task object: it never calls check_success() or anything that changes the
task's own success bookkeeping. Result: the number of completed stages, sent to the
policy with each observation (obs["oracle"]["stages_done"]).

swap_blocks (envs/swap_blocks.py, 3 moves through the empty tray, then press the button):
    move 1 done: block1 in the empty tray        (with_block_basket_idx_1 -> empty_basket_idx)
    move 2 done: block2 in block1's start tray   (with_block_basket_idx_2 -> with_block_basket_idx_1)
    move 3 done: block1 in block2's start tray   (empty_basket_idx -> with_block_basket_idx_2)
  and, for each move, both grippers open and both arms back within HOME_TOL of their
  start pose. "In tray" uses the same tolerances as the task's check_success
  (|dx| < 0.04, |dy| < 0.06, z < 0.765). This is the same definition as the training
  boundaries (stages.swap_blocks_event_ends: release, then both arms home), so training
  and evaluation write memory at the same kind of moment. Stages are latched in order.
"""

from __future__ import annotations

import numpy as np

HOME_TOL = 0.02


def _in_tray(task_env, block, tray_idx) -> bool:
    b = task_env.baskets[tray_idx].get_pose().p
    p = block.get_pose().p
    return abs(b[0] - p[0]) < 0.04 and abs(b[1] - p[1]) < 0.06 and p[2] < 0.765


class SwapBlocksOracle:
    n_stages_written = 3

    def __init__(self, task_env):
        self.env = task_env
        self.home = {a: np.asarray(task_env.get_arm_pose(a)[:3], dtype=float) for a in ("left", "right")}
        self.done = 0

    def _arms_home(self) -> bool:
        e = self.env
        return (e.is_left_gripper_open() and e.is_right_gripper_open() and
                all(np.linalg.norm(np.asarray(e.get_arm_pose(a)[:3]) - self.home[a]) < HOME_TOL
                    for a in ("left", "right")))

    def _move_done(self, k: int) -> bool:
        e = self.env
        if k == 0:
            return _in_tray(e, e.block1, e.empty_basket_idx)
        if k == 1:
            return _in_tray(e, e.block2, e.with_block_basket_idx_1)
        return _in_tray(e, e.block1, e.with_block_basket_idx_2)

    def __call__(self) -> int:
        while self.done < self.n_stages_written and self._move_done(self.done) and self._arms_home():
            self.done += 1
        return self.done


ORACLES = {"swap_blocks": SwapBlocksOracle}


def make_oracle(task_name: str, task_env):
    cls = ORACLES.get(task_name)
    if cls is None:
        raise KeyError(f"no stage oracle for task {task_name!r} (have {sorted(ORACLES)})")
    return cls(task_env)
