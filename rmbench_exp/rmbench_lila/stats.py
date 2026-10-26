"""Normalization statistics for RMBench (plan step 4). Read-only: never deletes files.

Output JSON keeps the main project's container key (``robotwin2``) because the
main project's VLAWrapper reads exactly that key; the ``meta`` block records that
the numbers come from RMBench.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .episodes import EpisodeReader, EpisodeRef
from .schema import NORM_STATS_CONTAINER_KEY, STATE_GROUP, VectorLayout

PI = float(np.pi)


def _episode_stats(args) -> dict:
    from .transforms import apply_sequence, build_transforms, count_sign_flips
    ref, s_layout, a_layout, alignment, transform_names = args
    with EpisodeReader(ref.path) as r:
        act = r.read_action(a_layout, alignment)
        st_raw = r.read_vector(s_layout, STATE_GROUP)
    st = apply_sequence(build_transforms(transform_names, s_layout), st_raw)
    out = {"task": ref.task, "path": ref.path, "len": int(act.shape[0]),
           "quat_flips_raw": count_sign_flips(st_raw, s_layout),
           "quat_flips_used": count_sign_flips(st, s_layout),
           "act_min": act.min(0), "act_max": act.max(0),
           "st_min": st.min(0), "st_max": st.max(0),
           "act_sample": act[:: max(1, act.shape[0] // 200)],
           "st_sample": st[:: max(1, st.shape[0] // 200)]}
    issues = []
    for d in a_layout.angle_dims:
        if np.any(np.abs(act[:, d]) > PI):
            issues.append(f"action[{d}] outside ±π")
    for d in a_layout.gripper_dims:
        if act[:, d].min() < -1e-3 or act[:, d].max() > 1 + 1e-3:
            issues.append(f"action[{d}] (gripper) outside [0,1]")
    if not np.all(np.isfinite(act)) or not np.all(np.isfinite(st)):
        issues.append("non-finite values")
    out["issues"] = issues
    return out


def compute_stats(refs: list[EpisodeRef], s_layout: VectorLayout, a_layout: VectorLayout,
                  alignment: str, num_workers: int = 8, exclude_flagged: bool = False,
                  state_transforms=()) -> dict:
    t0 = time.time()
    jobs = [(r, s_layout, a_layout, alignment, list(state_transforms)) for r in refs]
    if num_workers > 1:
        with ProcessPoolExecutor(num_workers) as ex:
            results = list(ex.map(_episode_stats, jobs, chunksize=4))
    else:
        results = [_episode_stats(j) for j in jobs]

    flagged = [r for r in results if r["issues"]]
    used = [r for r in results if not (exclude_flagged and r["issues"])]

    def agg(key_min, key_max, rows):
        return (np.min([r[key_min] for r in rows], 0).tolist(),
                np.max([r[key_max] for r in rows], 0).tolist())

    a_min, a_max = agg("act_min", "act_max", used)
    s_min, s_max = agg("st_min", "st_max", used)
    a_q = np.concatenate([r["act_sample"] for r in used])
    s_q = np.concatenate([r["st_sample"] for r in used])

    per_task: dict[str, dict] = {}
    for task in sorted({r["task"] for r in used}):
        rows = [r for r in used if r["task"] == task]
        tmin, tmax = agg("act_min", "act_max", rows)
        lens = [r["len"] for r in rows]
        per_task[task] = {"episodes": len(rows), "action_min": tmin, "action_max": tmax,
                          "quat_flips_raw": int(sum(r["quat_flips_raw"] for r in rows)),
                          "quat_flips_used": int(sum(r["quat_flips_used"] for r in rows)),
                          "len_min": int(min(lens)), "len_mean": float(np.mean(lens)),
                          "len_max": int(max(lens))}

    return {
        NORM_STATS_CONTAINER_KEY: {
            "meta": {
                "dataset": "rmbench",
                "state_layout": s_layout.name, "action_layout": a_layout.name,
                "action_alignment": alignment,
                "state_transforms": list(state_transforms),
                "total_files": len(results), "valid_files": len(used),
                "flagged_files": len(flagged), "flagged_excluded": exclude_flagged,
                "num_tasks": len(per_task),
                "processing_time": round(time.time() - t0, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "action": {"min": a_min, "max": a_max, "dim": a_layout.dim,
                       "q01": np.quantile(a_q, 0.01, 0).tolist(), "q99": np.quantile(a_q, 0.99, 0).tolist()},
            "state": {"min": s_min, "max": s_max, "dim": s_layout.dim,
                      "q01": np.quantile(s_q, 0.01, 0).tolist(), "q99": np.quantile(s_q, 0.99, 0).tolist()},
            "per_task": per_task,
        },
        "flagged": [{"path": r["path"], "task": r["task"], "issues": r["issues"]} for r in flagged],
    }


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))
