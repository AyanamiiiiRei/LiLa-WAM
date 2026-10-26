"""State transforms applied identically in training (whole episode at once) and at
inference (one step at a time). Selected by ``rmbench.state_transforms`` (a list of names).

quat_continuity
    q and -q are the same orientation, and the simulator sometimes switches sign
    between consecutive frames (seen in battery_try and observe_and_pickup, step 07).
    Rule, per end-effector quaternion: the first frame of an episode takes the sign
    with w >= 0; every later frame takes the sign closest to the previous frame
    (dot product >= 0). Offline (dataset/stats) and online (policy) give the same
    result as long as the policy sees every step — which the RMBench eval loop does.

Adding a transform: subclass StateTransform, register it in TRANSFORMS, list its
name in the config. Nothing else changes.
"""

from __future__ import annotations

import numpy as np

from .schema import VectorLayout


def quaternion_slices(layout: VectorLayout) -> list[slice]:
    """[x, y, z, qw, qx, qy, qz] pose fields -> slices of their quaternion part."""
    out = []
    for field, sl in layout.slices():
        if field.h5_key.endswith("ee_poses") and field.dim == 7:
            out.append(slice(sl.start + 3, sl.start + 7))
    return out


class StateTransform:
    name = ""

    def __init__(self, layout: VectorLayout):
        self.layout = layout

    def apply_sequence(self, states: np.ndarray) -> np.ndarray:
        """(T, D) whole episode, frame 0 first."""
        raise NotImplementedError

    def reset(self) -> None:
        """Start of a new episode (online use)."""

    def step(self, state: np.ndarray) -> np.ndarray:
        """One new (D,) state (online use)."""
        raise NotImplementedError


class QuatContinuity(StateTransform):
    name = "quat_continuity"

    def __init__(self, layout):
        super().__init__(layout)
        self.slices = quaternion_slices(layout)
        self.prev: np.ndarray | None = None

    def _fix(self, cur: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        cur = cur.copy()
        for sl in self.slices:
            q = cur[sl]
            ref = prev[sl] if prev is not None else None
            flip = (q[0] < 0) if ref is None else (float(np.dot(q, ref)) < 0)
            if flip:
                cur[sl] = -q
        return cur

    def apply_sequence(self, states):
        out = np.array(states, dtype=np.float32, copy=True)
        for t in range(out.shape[0]):
            out[t] = self._fix(out[t], out[t - 1] if t > 0 else None)
        return out

    def reset(self):
        self.prev = None

    def step(self, state):
        fixed = self._fix(np.asarray(state, dtype=np.float32), self.prev)
        self.prev = fixed
        return fixed


TRANSFORMS = {QuatContinuity.name: QuatContinuity}


def build_transforms(names, layout: VectorLayout) -> list[StateTransform]:
    names = list(names or [])
    unknown = [n for n in names if n not in TRANSFORMS]
    if unknown:
        raise KeyError(f"unknown state transform(s) {unknown}; known: {sorted(TRANSFORMS)}")
    ts = [TRANSFORMS[n](layout) for n in names]
    for t in ts:
        if isinstance(t, QuatContinuity) and not t.slices:
            raise ValueError(f"quat_continuity needs a state layout with ee poses; {layout.name!r} has none")
    return ts


def apply_sequence(transforms: list[StateTransform], states: np.ndarray) -> np.ndarray:
    for t in transforms:
        states = t.apply_sequence(states)
    return states


def count_sign_flips(states: np.ndarray, layout: VectorLayout) -> int:
    """Consecutive-frame quaternion pairs with a negative dot product (diagnostic)."""
    n = 0
    for sl in quaternion_slices(layout):
        q = states[:, sl]
        n += int((np.sum(q[1:] * q[:-1], axis=1) < 0).sum())
    return n
