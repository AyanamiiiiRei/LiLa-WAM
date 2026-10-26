#!/usr/bin/env python
"""Plan step 2: inspect the real RMBench data BEFORE trusting any loader code.

Needs only h5py, numpy, opencv (no torch, no GPU, no DINOv3). Read-only.

Checks, for ``--per_task`` sampled episodes of every task:
  * HDF5 tree, shapes, dtypes; every field the schema relies on exists with the right dim
  * value ranges (joints within ±π, grippers within [0,1], quaternion norms ≈ 1)
  * action[t] == state[t+1] for every shared field
  * additional_info/frequency
  * image format (marked standard vs legacy JPEG), frame count == state length,
    decoded size; saves first/last frames and a side-by-side with the mp4 frame
    for a visual colour check
and, over ALL episodes: episode count and length distribution per task.

Usage:
  python scripts/step02_inspect_data.py --data_root /path/RMBench/data/demo_clean
Outputs: artifacts/inspect/report.md, report.json, images/
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import _common  # noqa: F401
import cv2
import h5py
import numpy as np

from rmbench_lila.codec import CODEC_SOURCE, has_rgb_marker
from rmbench_lila.episodes import EpisodeReader, scan_episodes
from rmbench_lila.schema import (ACTION_GROUP, STATE_GROUP, STATE_LAYOUTS, ACTION_LAYOUTS,
                                 camera_h5_name)
from rmbench_lila.tasks import TASKS


def h5_tree(f: h5py.File) -> list[str]:
    rows = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            rows.append(f"{name}  shape={obj.shape} dtype={obj.dtype}")
    f.visititems(visit)
    return rows


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    # cv2.imwrite expects BGR: convert ONLY for writing to disk
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def video_frame(video: Path, i: int) -> np.ndarray | None:
    if not video.is_file():
        return None
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, bgr = cap.read()
    cap.release()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None


def inspect_episode(ref, camera: str, img_dir: Path) -> dict:
    res: dict = {"task": ref.task, "episode_id": ref.episode_id, "path": ref.path, "problems": []}
    P = res["problems"]
    with EpisodeReader(ref.path) as r:
        f = r.f
        res["tree"] = h5_tree(f)
        res["frequency"] = r.frequency()

        # ---- fields required by every registered layout
        fields = {fl.h5_key: fl.dim for lay in list(STATE_LAYOUTS.values()) + list(ACTION_LAYOUTS.values())
                  for fl in lay.fields}
        ranges = {}
        for grp in (STATE_GROUP, ACTION_GROUP):
            for key, dim in fields.items():
                if grp not in f or key not in f[grp]:
                    P.append(f"missing {grp}/{key}")
                    continue
                a = np.asarray(f[grp][key][()], dtype=np.float64)
                a = a[:, None] if a.ndim == 1 else a
                if a.shape[1] != dim:
                    P.append(f"{grp}/{key} dim {a.shape[1]} != expected {dim}")
                ranges[f"{grp}/{key}"] = {"min": a.min(0).round(4).tolist(), "max": a.max(0).round(4).tolist(),
                                          "len": int(a.shape[0])}
                if "arm_joint" in key and np.any(np.abs(a) > np.pi):
                    P.append(f"{grp}/{key} has joints outside ±π")
                if "ee_joint" in key and (a.min() < -1e-3 or a.max() > 1 + 1e-3):
                    P.append(f"{grp}/{key} gripper outside [0,1]: [{a.min():.3f},{a.max():.3f}]")
                if key.endswith("ee_poses") and a.shape[1] == 7:
                    qn = np.linalg.norm(a[:, 3:], axis=1)
                    ranges[f"{grp}/{key}"]["quat_norm_range"] = [round(float(qn.min()), 4), round(float(qn.max()), 4)]
                    if np.abs(qn - 1).max() > 1e-2:
                        P.append(f"{grp}/{key}: dims 3:7 are not a unit quaternion (layout assumption wrong?)")
        res["ranges"] = ranges

        # ---- action[t] == state[t+1]
        align = {}
        for key in fields:
            if key in f.get(STATE_GROUP, {}) and key in f.get(ACTION_GROUP, {}):
                s = np.asarray(f[STATE_GROUP][key][()], dtype=np.float64)
                a = np.asarray(f[ACTION_GROUP][key][()], dtype=np.float64)
                align[key] = float(np.abs(a[:-1] - s[1:]).max()) if len(s) > 1 else None
                if align[key] is not None and align[key] > 1e-5:
                    P.append(f"action[t] != state[t+1] for {key} (max diff {align[key]:.2e})")
        res["action_eq_next_state_maxdiff"] = align

        # ---- images
        cam = camera_h5_name(camera)
        if not r.has_camera(camera):
            P.append(f"missing vision/{cam}")
        else:
            n_img = r.num_frames(camera)
            n_state = ranges.get(f"{STATE_GROUP}/left_arm_joint_states", {}).get("len")
            res["num_frames"] = n_img
            if n_state is not None and n_img != n_state:
                P.append(f"{n_img} frames but {n_state} state steps")
            raw0 = r.raw_frame(camera, 0)
            res["image_format"] = "standard(XPL-RGB1 marker)" if has_rgb_marker(raw0) else "legacy(channel-reversed)"
            first, last = r.read_frames(camera, [0, n_img - 1])
            res["image_shape"] = list(first.shape)
            stem = f"{ref.task}_ep{ref.episode_id}"
            save_rgb(img_dir / f"{stem}_first.png", first)
            save_rgb(img_dir / f"{stem}_last.png", last)
            vid = video_frame(ref.video_path, 0)
            if vid is not None:
                if vid.shape != first.shape:
                    vid = cv2.resize(vid, (first.shape[1], first.shape[0]))
                save_rgb(img_dir / f"{stem}_hdf5_vs_video.png", np.concatenate([first, vid], axis=1))
                res["hdf5_vs_video_mean_abs_diff"] = float(np.abs(first.astype(int) - vid.astype(int)).mean())
                # a channel swap shows up as a much larger diff for the swapped version
                res["hdf5_swapped_vs_video_mean_abs_diff"] = float(
                    np.abs(first[..., ::-1].astype(int) - vid.astype(int)).mean())
                if res["hdf5_swapped_vs_video_mean_abs_diff"] < res["hdf5_vs_video_mean_abs_diff"]:
                    P.append("decoded frame matches the video BETTER after an R/B swap: colour bug!")
    return res


def main():
    p = _common.parser(__doc__, config=False)
    p.add_argument("--data_root", required=True, help=".../RMBench/data/demo_clean")
    p.add_argument("--embodiment", default="aloha_agilex")
    p.add_argument("--camera", default="head_camera")
    p.add_argument("--per_task", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(_common.SUBPROJECT_ROOT / "artifacts" / "inspect"))
    args = p.parse_args()

    out = Path(args.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    refs = scan_episodes(args.data_root, args.embodiment)

    by_task = defaultdict(list)
    for r in refs:
        by_task[r.task].append(r)

    # ---- length distribution over all episodes
    lengths = {}
    for task, eps in sorted(by_task.items()):
        ls = []
        for r in eps:
            with h5py.File(r.path, "r") as f:
                ls.append(int(f[STATE_GROUP]["left_arm_joint_states"].shape[0]))
        ls = np.array(ls)
        lengths[task] = {"episodes": len(ls), "min": int(ls.min()), "p50": float(np.median(ls)),
                         "mean": float(ls.mean()), "max": int(ls.max()), "total_frames": int(ls.sum()),
                         "kind": TASKS[task].kind if task in TASKS else "unknown"}

    # ---- detailed checks on a sample
    rng = random.Random(args.seed)
    details = []
    for task, eps in sorted(by_task.items()):
        for ref in rng.sample(eps, min(args.per_task, len(eps))):
            print(f"inspecting {task} ep{ref.episode_id}")
            details.append(inspect_episode(ref, args.camera, img_dir))

    report = {"data_root": args.data_root, "codec": CODEC_SOURCE, "lengths": lengths, "episodes": details,
              "unknown_tasks": sorted(set(by_task) - set(TASKS)),
              "missing_tasks": sorted(set(TASKS) - set(by_task))}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # ---- markdown summary
    L = ["# Step 02 — RMBench data inspection", "", f"data_root: `{args.data_root}`  codec: `{CODEC_SOURCE}`", ""]
    L += ["## Episodes and lengths (all episodes)", "",
          "| task | type | episodes | min | median | mean | max | total frames |", "|---|---|---|---|---|---|---|---|"]
    for t, d in lengths.items():
        L.append(f"| {t} | {d['kind']} | {d['episodes']} | {d['min']} | {d['p50']:.0f} | {d['mean']:.0f} | {d['max']} | {d['total_frames']} |")
    L += ["", f"Tasks in data but not in rmbench_lila/tasks.py: {report['unknown_tasks'] or 'none'}",
          f"Tasks in tasks.py but not in data: {report['missing_tasks'] or 'none'}", "",
          "## Sampled episodes", ""]
    for d in details:
        L.append(f"### {d['task']} / episode {d['episode_id']}")
        L.append(f"- frequency: {d.get('frequency')}  frames: {d.get('num_frames')}  "
                 f"image: {d.get('image_format')} {d.get('image_shape')}")
        if "hdf5_vs_video_mean_abs_diff" in d:
            L.append(f"- |hdf5 - video| = {d['hdf5_vs_video_mean_abs_diff']:.1f}, "
                     f"|swapped - video| = {d['hdf5_swapped_vs_video_mean_abs_diff']:.1f} (first should be smaller)")
        L.append(f"- action[t]==state[t+1] max diff: " +
                 ", ".join(f"{k}={v:.1e}" for k, v in d["action_eq_next_state_maxdiff"].items() if v is not None))
        L.append("- problems: " + ("**" + "; ".join(d["problems"]) + "**" if d["problems"] else "none"))
        L.append("")
    L += ["## HDF5 tree of the first sampled episode", "", "```", *details[0]["tree"], "```", "",
          "## Still to do by hand", "",
          "- Look at `images/*_hdf5_vs_video.png`: left (HDF5) and right (mp4) must have the same colours.",
          "- Read `envs/<task>.py` for every task and confirm/correct the memory notes in "
          "`rmbench_lila/tasks.py` (set `verified=True`)."]
    (out / "report.md").write_text("\n".join(L))
    n_prob = sum(bool(d["problems"]) for d in details)
    print(f"\nWrote {out/'report.md'} — {len(details)} episodes inspected, {n_prob} with problems.")


if __name__ == "__main__":
    main()
