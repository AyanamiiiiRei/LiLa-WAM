"""quat_continuity: offline (training) and online (policy) must agree exactly,
and sign flips must disappear without changing the represented orientations.
Needs only numpy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rmbench_lila.schema import STATE_EE16, STATE_JOINT14  # noqa: E402
from rmbench_lila.transforms import (apply_sequence, build_transforms,  # noqa: E402
                                     count_sign_flips, quaternion_slices)


def _episode(T=200, seed=0):
    """ee16 states whose quaternions rotate smoothly but are stored with random sign flips."""
    rng = np.random.default_rng(seed)
    s = np.zeros((T, 16), np.float32)
    for sl, phase in zip(quaternion_slices(STATE_EE16), (0.0, 1.0)):
        ang = np.linspace(0, 3.0, T) + phase                 # passes through w = cos(ang/2) < 0
        q = np.stack([np.cos(ang / 2), np.sin(ang / 2), np.zeros(T), np.zeros(T)], 1)
        q *= rng.choice([-1.0, 1.0], size=(T, 1))            # simulator-style random sign
        s[:, sl] = q
    s[:, 7] = rng.uniform(0, 1, T)                             # grippers etc. must stay untouched
    s[:, 15] = rng.uniform(0, 1, T)
    return s


def test_slices():
    assert [(s.start, s.stop) for s in quaternion_slices(STATE_EE16)] == [(3, 7), (11, 15)]
    assert quaternion_slices(STATE_JOINT14) == []


def test_removes_flips_keeps_orientation_and_other_dims():
    raw = _episode()
    assert count_sign_flips(raw, STATE_EE16) > 10
    out = apply_sequence(build_transforms(["quat_continuity"], STATE_EE16), raw)
    assert count_sign_flips(out, STATE_EE16) == 0
    for sl in quaternion_slices(STATE_EE16):
        assert out[0, sl][0] >= 0                                       # first frame: w >= 0
        assert np.allclose(np.abs(out[:, sl]), np.abs(raw[:, sl]))       # same orientation (q or -q)
    other = [i for i in range(16) if not any(sl.start <= i < sl.stop for sl in quaternion_slices(STATE_EE16))]
    assert np.array_equal(out[:, other], raw[:, other])


def test_online_equals_offline():
    raw = _episode(seed=3)
    off = apply_sequence(build_transforms(["quat_continuity"], STATE_EE16), raw)
    (t,) = build_transforms(["quat_continuity"], STATE_EE16)
    for _ in range(2):                                                    # reset() must fully restart
        t.reset()
        on = np.stack([t.step(x) for x in raw])
        assert np.array_equal(on, off)


def test_rejects_layout_without_poses():
    try:
        build_transforms(["quat_continuity"], STATE_JOINT14)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed", name)
