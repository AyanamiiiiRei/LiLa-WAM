"""Phase-2 triggers: the online proprio rule must give the same stage ends as the offline
training-boundary rule (stages.swap_blocks_event_ends) on the same trajectory."""

import numpy as np

from rmbench_lila.stages import swap_blocks_event_ends
from rmbench_lila.triggers import ProprioTrigger, TimeTrigger


def _traj():
    T = 300
    L = np.zeros((T, 3)); R = np.zeros((T, 3)); gl = np.ones(T); gr = np.ones(T)
    for a, b, arm, g in ((10, 40, R, gr), (80, 110, R, gr), (150, 180, L, gl)):
        arm[a:b + 20] = 0.1
        g[a + 5:b] = 0.0
    return L, R, gl, gr


def _state(L, R, gl, gr, t):
    s = np.zeros(16, np.float32)
    s[0:3], s[7], s[8:11], s[15] = L[t], gl[t], R[t], gr[t]
    return s


def test_proprio_matches_offline_rule():
    L, R, gl, gr = _traj()
    ends = swap_blocks_event_ends(L, R, gl, gr)
    trig, first = ProprioTrigger(3), {}
    for t in range(len(L)):
        d = trig.update(_state(L, R, gl, gr, t))
        first.setdefault(d, t)
    assert [first[k] for k in (1, 2, 3)] == ends


def test_proprio_needs_a_grasp_and_caps():
    L, R, gl, gr = _traj()
    trig = ProprioTrigger(3)
    assert trig.update(_state(L, R, gl, gr, 0)) == 0      # at home, never grasped: no write
    for t in range(len(L)):
        trig.update(_state(L, R, gl, gr, t))
    for _ in range(3):                                    # extra grasp/release cycles: capped at 3
        trig.update(np.r_[np.zeros(7), 0.0, np.zeros(7), 1.0])
        trig.update(np.r_[np.zeros(7), 1.0, np.zeros(7), 1.0])
    assert trig.done == 3


def test_time_trigger():
    tt = TimeTrigger([176, 352, 528])
    assert [tt.stages_done(t) for t in (0, 175, 176, 400, 600)] == [0, 0, 1, 2, 3]
