"""Failure diagnostics from the recorded policy-view states (plan step 11, automatic part).

Reads <ckpt_dir>/episodes.jsonl and <ckpt_dir>/policy_view/<task>_seed<seed>.npz
(written by the XPolicyLab adapter: states = ee16 per env step, actions = executed joint14).
Everything here is derived from the robot's own end-effector trajectory, so it can only
answer questions about WHERE the robot went; whether it held the right object still needs
the videos (analysis/failure_labels.py).

  python analysis/state_diagnostics.py runs/baseline-seed0/rmbench_eval/ckpt_85928

ee16 layout: [0:3] left xyz, [3:7] left quat, [7] left gripper (1 open, 0 closed),
             [8:11] right xyz, [11:15] right quat, [15] right gripper.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

# put_back_block (envs/put_back_block.py): 4 mats + centre, table xy in metres
PBB_MATS = {"mat(0.2,-0.1)": (0.2, -0.1), "mat(0.1,0.0)": (0.1, 0.0),
            "mat(0.0,-0.1)": (0.0, -0.1), "mat(0.1,-0.2)": (0.1, -0.2)}
PBB_CENTRE = (0.1, -0.1)
PBB_DEMO_PRESS_Z = 0.926          # left EE z at the button press in the demonstrations


def _edges(mask: np.ndarray) -> list[int]:
    return [t for t in range(1, len(mask)) if mask[t] and not mask[t - 1]]


def _falling(mask: np.ndarray) -> list[int]:
    return [t for t in range(1, len(mask)) if not mask[t] and mask[t - 1]]


def _nearest(p, places: dict, tol=0.04) -> str:
    k = min(places, key=lambda k: np.hypot(p[0] - places[k][0], p[1] - places[k][1]))
    return k if np.hypot(p[0] - places[k][0], p[1] - places[k][1]) < tol else "other"


def load(ckpt_dir: Path):
    eps = {}
    for line in open(ckpt_dir / "episodes.jsonl"):
        if line.strip():
            e = json.loads(line)
            eps[(e["task"], e["seed"])] = e          # last record per (task, seed)
    for (task, seed), e in sorted(eps.items()):
        f = ckpt_dir / "policy_view" / f"{task}_seed{seed}.npz"
        if f.exists():
            s = np.load(f)["states"][: e["env_steps"]]
            yield task, seed, e, s


def put_back_block(items) -> dict:
    places = dict(PBB_MATS, centre=PBB_CENTRE)
    out = collections.Counter()
    final = collections.Counter()
    press = []
    for _, _, e, s in items:
        closed = s[:, 15] < 0.5
        grasps, releases = _edges(closed), _falling(closed)
        press.append(float(s[:, 2].min()))
        if not grasps or not releases:
            out["no grasp/release"] += 1
            continue
        start = _nearest(s[grasps[0], 8:10], places)
        first_rel = _nearest(s[releases[0], 8:10], places)
        last_rel = _nearest(s[releases[-1], 8:10], places)
        out["phase1 to centre"] += first_rel == "centre"
        out["returned to own mat"] += last_rel == start
        out["n"] += 1
        final[last_rel] += 1
    press = np.array(press)
    return {"counts": dict(out), "final_release_location": dict(final),
            "press_z_median": round(float(np.median(press)), 4),
            "press_z_min": round(float(press.min()), 4),
            "press_as_deep_as_demo(<=demo+3mm)": int((press <= PBB_DEMO_PRESS_Z + 0.003).sum()),
            "chance_return": 0.25}


def observe_and_pickup(items) -> dict:
    """Moving > 3 cm during the first 20 obs fails the episode (envs/observe_and_pickup.py)."""
    c = collections.Counter()
    first_move = []
    for _, _, e, s in items:
        dL = np.linalg.norm(s[:, 0:3] - s[0, 0:3], axis=1)
        dR = np.linalg.norm(s[:, 8:11] - s[0, 8:11], axis=1)
        moved = np.where((dL > 0.03) | (dR > 0.03))[0]
        first_move.append(int(moved[0]) if len(moved) else None)
        gL, gR = (s[:, 7] < 0.5).any(), (s[:, 15] < 0.5).any()
        if e["success"]:
            c["success"] += 1
        elif len(moved) and moved[0] < 20:
            c["moved before wall (auto-fail)"] += 1
        elif not len(moved):
            c["never moved"] += 1
        else:
            c[f"grasped with {'both' if gL and gR else 'left' if gL else 'right' if gR else 'no'} arm"] += 1
    fm = [x for x in first_move if x is not None]
    return {"counts": dict(c), "first_move_step_median": float(np.median(fm)) if fm else None,
            "chance_random_object": 0.2}


def grasp_counts(items) -> dict:
    """Generic: grasps per arm and longest idle stretch, split by success."""
    rows = {True: [], False: []}
    for _, _, e, s in items:
        v = np.linalg.norm(np.diff(s[:, [0, 1, 2, 8, 9, 10]], axis=0), axis=1)
        run = best = 0
        for idle in v < 5e-4:
            run = run + 1 if idle else 0
            best = max(best, run)
        rows[e["success"]].append((len(_edges(s[:, 7] < 0.5)), len(_edges(s[:, 15] < 0.5)), best))
    res = {}
    for ok, r in rows.items():
        if r:
            a = np.array(r)
            res["success" if ok else "fail"] = {"n": len(r), "grasps_L_median": float(np.median(a[:, 0])),
                                                "grasps_R_median": float(np.median(a[:, 1])),
                                                "longest_idle_steps_max": int(a[:, 2].max())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", type=Path)
    args = ap.parse_args()
    by_task = collections.defaultdict(list)
    for item in load(args.ckpt_dir):
        by_task[item[0]].append(item)
    report = {}
    for task, items in sorted(by_task.items()):
        report[task] = {"generic": grasp_counts(items)}
        if task == "put_back_block":
            report[task]["put_back_block"] = put_back_block(items)
        if task == "observe_and_pickup":
            report[task]["observe_and_pickup"] = observe_and_pickup(items)
    out = args.ckpt_dir / "state_diagnostics.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
