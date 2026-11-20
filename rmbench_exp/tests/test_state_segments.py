"""State segmentation rules (rmbench_lila/state_segments.py) on synthetic ee16 trajectories.

  python -m pytest tests/test_state_segments.py -q
"""
import numpy as np

from rmbench_lila.state_segments import segment


def _traj(n):
    s = np.zeros((n, 16))
    s[:, 0:3] = [-0.3, -0.1, 0.94]
    s[:, 8:11] = [0.3, -0.1, 0.94]
    s[:, 7] = s[:, 15] = 1.0          # open
    return s


def test_grasp_place_is_release_not_press():
    s = _traj(120)
    s[10:30, 2] = np.linspace(0.94, 0.90, 20)        # descend open
    s[30:60, 7] = 0.0                                 # close at the bottom, carry
    s[30:50, 2] = np.linspace(0.90, 0.96, 20)
    s[50:60, 2] = np.linspace(0.96, 0.92, 10)        # lower closed, open at the bottom
    s[60:80, 2] = np.linspace(0.92, 0.95, 20)
    assert segment(s, "cover_blocks") == [60]
    assert segment(s, "press_button") == []


def test_press_with_closed_gripper():
    s = _traj(200)
    s[5:, 7] = 0.0                                    # closed before descending
    for k, t0 in enumerate((30, 90)):
        s[t0:t0 + 10, 2] = np.linspace(0.94, 0.91, 10)
        s[t0 + 10:t0 + 20, 2] = np.linspace(0.91, 0.94, 10)
    ev = segment(s, "press_button")
    assert len(ev) == 2 and 39 <= ev[0] <= 45 and 99 <= ev[1] <= 105


def test_home_all_waits_for_both_arms():
    s = _traj(200)
    s[20:120, 0] += 0.1                               # left away
    s[20:160, 8] -= 0.1                               # right away longer
    s[40:60, 7] = s[40:60, 15] = 0.0                 # both grasp and release
    assert segment(s, "battery_try") == [160]


def test_observe_move_start_once():
    s = _traj(100)
    s[60:, 0] += np.linspace(0, 0.05, 40)
    ev = segment(s, "observe_and_pickup")
    assert len(ev) == 1 and 60 <= ev[0] <= 66
