#!/usr/bin/env python
"""Stage-memory experiment: check the evaluation oracle on the EXPERT demos (plan checklist:
"in expert replay, oracle boundary vs annotation boundary differ by at most 1 action chunk").

Run in the RMBench conda env, FROM THE RMBENCH ROOT (RMBench uses relative paths):

  cd /path/to/RMBench
  python /path/to/rmbench_exp/scripts/step26_oracle_replay_check.py \
      --boundaries /path/to/rmbench_exp/artifacts/stage_memory/swap_blocks/boundaries_gt.json \
      [--episodes 0-49] [--task swap_blocks] [--task_config demo_clean]

What it does (RMBench files are not edited; nothing is written into data/):
  * For each demo episode, rebuilds the scene with the demo's seed (seed.txt) and replays the
    demo's saved joint paths (_traj_data/episodeN.pkl), exactly like the data-collection pass of
    scripts/collect_data.py; if the path file is missing it re-plans (same seed, deterministic).
  * Base_Task._take_picture (the function that saves one HDF5 frame) is replaced IN THIS PROCESS
    by a function that, instead of saving, records for that frame index: the oracle's
    stages_done (rmbench_lila/oracle.py, the same object the eval uses; created at frame 0 like
    in the eval), both EE positions and both gripper values.
  * Compares, per episode and per move k = 1..3:
      oracle frame f_k   first frame at which stages_done reaches k
      train  end  r_k    swap_blocks_event_ends on the HDF5 state (the definition used to train)
      grid   end  g_k    boundaries_gt.json (r_k rounded up to the 16-frame grid; where the slot
                         is written in training)
      annotation end a_k nearest end of a language-annotation segment (one move() call)
    and checks that the replay reproduces the demo (frame count, max EE distance to the HDF5).
Pass (per episode): replay aligned, oracle reaches 3, |ceil16(f_k) - g_k| <= 16 and
|f_k - a_k| <= 16 for every k. Report: <out>/oracle_replay.md and .json.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

SUB = Path(__file__).resolve().parents[1]
GRID = 16


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, SUB / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oracle_mod = _load("lila_stage_oracle", "rmbench_lila/oracle.py")
stages_mod = _load("lila_stages", "rmbench_lila/stages.py")


def build_args(task, task_config):
    """Same task arguments as scripts/collect_data.py main() + its data-collection pass."""
    import yaml
    from envs import CONFIGS_PATH
    with open(os.path.join(CONFIGS_PATH, f"{task_config}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    args["task_name"] = task
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        emb = yaml.safe_load(f)
    et = args["embodiment"]

    def robot_file(e):
        return emb[e]["file_path"]

    def emb_cfg(rf):
        with open(os.path.join(rf, "config.yml"), encoding="utf-8") as f:
            return yaml.safe_load(f)
    if len(et) == 1:
        args["left_robot_file"] = args["right_robot_file"] = robot_file(et[0])
        args["dual_arm_embodied"] = True
        name = str(et[0])
    else:
        args["left_robot_file"], args["right_robot_file"] = robot_file(et[0]), robot_file(et[1])
        args["embodiment_dis"] = et[2]
        args["dual_arm_embodied"] = False
        name = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = emb_cfg(args["left_robot_file"])
    args["right_embodiment_config"] = emb_cfg(args["right_robot_file"])
    args["embodiment_name"] = name
    args["task_config"] = task_config
    import re
    args["save_path"] = os.path.join(args["save_path"], task_config, task,
                                     re.sub(r"[^A-Za-z0-9_]+", "_", name).lower())
    args["render_freq"] = 0
    args["save_data"] = False          # nothing is written; frames are recorded by the hook
    return args


REC: dict = {}


def install_hook():
    from envs import _base_task

    def _take_picture(self):
        self.language_annotation_cache += 1
        if self.FRAME_IDX == 0:
            REC["oracle"] = oracle_mod.make_oracle(type(self).__name__, self)
            REC["frames"] = []
        REC["frames"].append({
            "stages_done": int(REC["oracle"]()),
            "l": np.asarray(self.get_arm_pose("left")[:3], float).tolist(),
            "r": np.asarray(self.get_arm_pose("right")[:3], float).tolist(),
            "lg": float(self.robot.get_left_gripper_val()),
            "rg": float(self.robot.get_right_gripper_val()),
        })
        self.FRAME_IDX += 1
    _base_task.Base_Task._take_picture = _take_picture


def hdf5_state(path):
    import h5py
    with h5py.File(path, "r") as f:
        return {k: f[f"state/{k}"][()] for k in
                ("left_ee_poses", "right_ee_poses", "left_ee_joint_states", "right_ee_joint_states")}


def annotation_ends(ann_entry):
    """End frame of every annotation segment (each segment = one move() call)."""
    return [int(c) - 1 for c in np.cumsum([int(n) for _, n in ann_entry])]


def replay(task_env, args, idx, seed, traj_dir):
    traj = traj_dir / f"episode{idx}.pkl"
    a = dict(args)
    if traj.is_file():
        import pickle
        with open(traj, "rb") as f:
            d = pickle.load(f)
        a.update(need_plan=False, left_joint_path=d["left_joint_path"], right_joint_path=d["right_joint_path"])
        mode = "saved path"
    else:
        a.update(need_plan=True)
        mode = "re-planned"
    REC.clear()
    task_env.setup_demo(now_ep_num=idx, seed=seed, **a)
    if not a["need_plan"]:
        task_env.set_path_lst(a)
    task_env.play_once()
    ok = bool(task_env.plan_success and task_env.check_success())
    task_env.close_env(clear_cache=((idx + 1) % 5 == 0))
    return REC.get("frames", []), mode, ok


def first_reach(stages_done, k):
    hit = [i for i, s in enumerate(stages_done) if s >= k]
    return hit[0] if hit else None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--boundaries", required=True)
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--task_config", default="demo_clean")
    p.add_argument("--episodes", default=None, help="e.g. 0-49 or 0,3,7 (default: all in boundaries)")
    p.add_argument("--out", default=None, help="default: folder of --boundaries")
    args_cli = p.parse_args()

    if not Path("envs/_base_task.py").is_file():
        raise SystemExit("run this from the RMBench root (the folder that contains envs/)")
    sys.path.insert(0, os.getcwd())
    install_hook()
    args = build_args(args_cli.task, args_cli.task_config)
    root = Path(args["save_path"])
    seeds = [int(s) for s in (root / "seed.txt").read_text().split()]
    ann = json.loads((root / "language_annotation.json").read_text())
    bounds = json.loads(Path(args_cli.boundaries).read_text())
    out = Path(args_cli.out or Path(args_cli.boundaries).parent)
    if args_cli.episodes:
        if "-" in args_cli.episodes:
            a, b = map(int, args_cli.episodes.split("-"))
            ids = list(range(a, b + 1))
        else:
            ids = [int(x) for x in args_cli.episodes.split(",")]
    else:
        ids = sorted(int(k.split("_")[1].split(".")[0]) for k in bounds)

    task_env = getattr(importlib.import_module(f"envs.{args_cli.task}"), args_cli.task)()
    rows = []
    for idx in ids:
        name = f"episode_{idx:07d}.hdf5"
        row = {"episode": name, "seed": seeds[idx]}
        try:
            frames, mode, ok = replay(task_env, args, idx, seeds[idx], root / "_traj_data")
            st = hdf5_state(root / "data" / name)
            n_h, n_r = len(st["left_ee_poses"]), len(frames)
            m = min(n_h, n_r)
            L = np.array([f["l"] for f in frames[:m]]); R = np.array([f["r"] for f in frames[:m]])
            ee_err = float(max(np.abs(L - st["left_ee_poses"][:m, :3]).max(),
                               np.abs(R - st["right_ee_poses"][:m, :3]).max()))
            train_ends = stages_mod.swap_blocks_event_ends(
                st["left_ee_poses"][:, :3], st["right_ee_poses"][:, :3],
                st["left_ee_joint_states"].reshape(-1), st["right_ee_joint_states"].reshape(-1))
            grid_ends = [s[1] for s in bounds[name][:3]]
            a_ends = annotation_ends(ann[f"episode_{idx}"])
            sd = [f["stages_done"] for f in frames]
            orc = [first_reach(sd, k) for k in (1, 2, 3)]
            d_grid = [None if f is None else int(np.ceil(f / GRID) * GRID) - g for f, g in zip(orc, grid_ends)]
            d_train = [None if f is None else f - r for f, r in zip(orc, train_ends)]
            d_ann = [None if f is None else min((f - a for a in a_ends), key=abs) for f in orc]
            aligned = 0 <= n_r - n_h <= 1 and ee_err < 0.005   # the HDF5 may drop the last frame
            passed = (aligned and None not in orc and all(abs(x) <= GRID for x in d_grid)
                      and all(abs(x) <= GRID for x in d_ann))
            row.update(mode=mode, replay_success=ok, frames_hdf5=n_h, frames_replay=n_r,
                       max_ee_err_m=round(ee_err, 5), aligned=aligned, oracle=orc, train=train_ends,
                       grid=grid_ends, d_train=d_train, d_grid=d_grid, d_ann=d_ann, passed=passed)
        except Exception as e:                       # one bad episode must not stop the check
            row.update(error=f"{type(e).__name__}: {e}", passed=False)
            traceback.print_exc()
        rows.append(row)
        print(json.dumps(row))

    (out / "oracle_replay.json").write_text(json.dumps(rows, indent=1))
    good = [r for r in rows if "error" not in r]

    def col(key, k):
        return [r[key][k] for r in good if r[key][k] is not None]
    lines = [f"# Oracle vs boundaries on expert replay ({args_cli.task}, {len(rows)} episodes)", "",
             f"passed: {sum(r['passed'] for r in rows)}/{len(rows)}; replay aligned with HDF5: "
             f"{sum(r.get('aligned', False) for r in rows)}/{len(rows)}; errors: {len(rows) - len(good)}", "",
             "differences in frames (1 action chunk = 16 frames); min / median / max over episodes", "",
             "| move | oracle - train end | ceil16(oracle) - grid end | oracle - nearest annotation end | oracle never reached |",
             "|---|---|---|---|---|"]
    for k in range(3):
        cells = []
        for key in ("d_train", "d_grid", "d_ann"):
            v = col(key, k)
            cells.append(f"{min(v)} / {int(np.median(v))} / {max(v)}" if v else "-")
        miss = sum(r["oracle"][k] is None for r in good)
        lines.append(f"| {k + 1} | {cells[0]} | {cells[1]} | {cells[2]} | {miss} |")
    bad = [r for r in rows if not r["passed"]]
    if bad:
        lines += ["", "## Episodes that did not pass", ""]
        lines += [f"- {r['episode']}: " + (r.get("error") or
                  f"aligned={r['aligned']} (frames {r['frames_replay']}/{r['frames_hdf5']}, ee err "
                  f"{r['max_ee_err_m']} m), oracle {r['oracle']}, grid {r['grid']}, d_ann {r['d_ann']}")
                  for r in bad]
    (out / "oracle_replay.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten: {out / 'oracle_replay.md'}, {out / 'oracle_replay.json'}")


if __name__ == "__main__":
    main()
