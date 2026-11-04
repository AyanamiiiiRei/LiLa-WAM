"""Stage-memory experiment, phase 2: what decides WHEN a memory slot is written at test time.

Phase 1 used the simulator oracle (rmbench_lila/oracle.py, privileged state). Phase 2 keeps
the SAME M checkpoint and only replaces the trigger (docs/STAGE_MEMORY.md, "第 2 期"):

  oracle   obs["oracle"]["stages_done"]  (privileged; phase-1 reference)
  proprio  ProprioTrigger: the training-boundary rule run online on the robot state the
           policy already receives: after a gripper has been closed and opened again, the
           move is done once BOTH end-effectors are within home_tol of their episode-start
           position. No simulator state; cannot check that the block is in a tray.
  head     SwitchHead: a small MLP on what the model sees (current-frame CLS, foresight local
           change, number of slots written) -> P(current stage has ended). Trained offline on
           the demos (scripts/step28/step29).
  time     TimeTrigger: writes at fixed env steps (mean demo stage ends); ignores the scene.
           Baseline: offline it matches 84% of the demo boundaries (+-1 chunk) because the
           scripted demos have regular timing, so it shows what the closed loop adds.

Every trigger returns "stages done so far" (monotone, <= n_stages), which is what
OnlineStageMemory.update() consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

TRIGGERS = ("oracle", "proprio", "head", "time")


class ProprioTrigger:
    """Online version of stages.swap_blocks_event_ends on the ee16 state
    (left pose 0:7, left gripper 7, right pose 8:15, right gripper 15; gripper 1 = open)."""

    def __init__(self, n_stages: int = 3, home_tol: float = 0.02, grip_open: float = 0.5):
        self.n_stages, self.home_tol, self.grip_open = n_stages, home_tol, grip_open
        self.reset()

    def reset(self) -> None:
        self.home = None
        self.armed = False          # a gripper has been closed since the last write
        self.done = 0

    def update(self, state: np.ndarray) -> int:
        s = np.asarray(state, np.float32).reshape(-1)
        left, right = s[0:3], s[8:11]
        if self.home is None:
            self.home = (left.copy(), right.copy())
        closed = s[7] <= self.grip_open or s[15] <= self.grip_open
        if closed:
            self.armed = True
        elif self.armed and self.done < self.n_stages and \
                np.linalg.norm(left - self.home[0]) < self.home_tol and \
                np.linalg.norm(right - self.home[1]) < self.home_tol:
            self.done += 1
            self.armed = False
        return self.done


class TimeTrigger:
    def __init__(self, ends_steps):
        self.ends = [int(e) for e in ends_steps]

    def stages_done(self, env_step: int) -> int:
        return sum(env_step >= e for e in self.ends)


# ----------------------------------------------------------------- learned head
def head_input(cls: np.ndarray, delta: np.ndarray, k: int, n_stages: int = 3) -> np.ndarray:
    onehot = np.zeros(n_stages, np.float32)
    onehot[min(int(k), n_stages - 1)] = 1.0
    return np.concatenate([np.asarray(cls, np.float32).reshape(-1),
                           np.asarray(delta, np.float32).reshape(-1), onehot])


def build_head_mlp(in_dim: int, hidden: int = 256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, 1))


class SwitchHead:
    """P(the current stage has ended | CLS_t, foresight change_t, k slots written).
    Inputs are standardised with the training-set mean/std stored in the checkpoint."""

    needs_vision = True

    def __init__(self, path: str | Path, threshold: float | None = None):
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.net = build_head_mlp(int(ck["in_dim"]), int(ck["hidden"]))
        self.net.load_state_dict(ck["state_dict"])
        self.net.eval()
        self.mean, self.std = ck["mean"].astype(np.float32), ck["std"].astype(np.float32)
        self.n_stages = int(ck["n_stages"])
        self.threshold = float(ck["threshold"] if threshold is None else threshold)
        self.path = str(path)
        self.log: list = []          # (chunk, k, p) per decision, for the episode record

    def reset(self) -> None:
        self.log = []

    def prob(self, cls, delta, k) -> float:
        import torch
        x = (head_input(cls, delta, k, self.n_stages) - self.mean) / self.std
        with torch.no_grad():
            return float(torch.sigmoid(self.net(torch.from_numpy(x)[None]))[0, 0])

    def decide(self, cls, delta, k: int, chunk: int) -> int:
        """Stages done after this chunk's decision (at most one new write per chunk)."""
        if k >= self.n_stages:
            return k
        p = self.prob(cls, delta, k)
        self.log.append((chunk, k, round(p, 3)))
        return k + 1 if p > self.threshold else k
