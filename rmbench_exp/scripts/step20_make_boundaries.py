#!/usr/bin/env python
"""Stage-memory experiment, step 1: stage boundaries from RMBench's language annotations.

  python scripts/step20_make_boundaries.py --task put_back_block
  -> artifacts/stage_memory/<task>/boundaries_gt.json  (+ report.md, check frames)

For every episode: merge consecutive identical annotation segments, take each segment's
last frame, round it UP to the 16-frame grid (the last stage runs to the end of the
episode). The report compares the raw ends with events read from the robot state
(put_back_block: right gripper releases the block at the centre = end of "move to centre";
left EE lowest point = button press), so wrong boundaries show up as outliers.
Writes frames at each boundary for a visual check (--frames N episodes).
"""

from __future__ import annotations

import json
from pathlib import Path

import _common
import numpy as np

from rmbench_lila.episodes import EpisodeReader, scan_episodes
from rmbench_lila.stages import GRID, grid_stages, merge_annotation, raw_ends


def state_events(reader: EpisodeReader) -> dict:
    import h5py  # noqa: F401
    f = reader.f
    rg = f["state/right_ee_joint_states"][()].reshape(-1)
    lz = f["state/left_ee_poses"][()][:, 2]
    rel = [t for t in range(1, len(rg)) if rg[t] > 0.5 >= rg[t - 1]]
    return {"right_release": rel[0] if rel else None, "left_z_min": int(np.argmin(lz))}


def main():
    p = _common.parser(__doc__)
    p.add_argument("--task", default="put_back_block")
    p.add_argument("--out", default=None)
    p.add_argument("--frames", type=int, default=5, help="episodes to dump boundary frames for")
    args = p.parse_args()
    cfg = _common.load(args)
    root = Path(cfg.rmbench.data_root)
    ann_path = root / args.task / cfg.rmbench.embodiment / "language_annotation.json"
    ann = json.loads(ann_path.read_text())
    out = Path(args.out or f"artifacts/stage_memory/{args.task}")
    out.mkdir(parents=True, exist_ok=True)

    refs = scan_episodes(cfg.rmbench.data_root, cfg.rmbench.embodiment, [args.task])
    boundaries, rows, names = {}, [], None
    for ref in refs:
        key = f"episode_{int(ref.episode_id)}"
        merged = merge_annotation(ann[key])
        if names is None:
            names = [d for d, _ in merged]
        if len(merged) != len(names):
            raise ValueError(f"{key}: {len(merged)} segments, expected {len(names)}: {merged}")
        ends = raw_ends(merged)
        with EpisodeReader(ref.path) as r:
            n = r.num_frames(list(cfg.dataset.camera_names)[0])
            ev = state_events(r)
        stages = grid_stages(ends, n)
        boundaries[Path(ref.path).name] = stages
        rows.append({"episode": Path(ref.path).name, "length": n, "raw_ends": ends,
                     "grid_ends": [s[1] for s in stages], **ev,
                     "ann_total": int(sum(k for _, k in merged))})

    (out / "boundaries_gt.json").write_text(json.dumps(boundaries, indent=1))

    # ---- report
    def stat(xs):
        xs = np.asarray([x for x in xs if x is not None])
        return f"median {np.median(xs):.0f}, min {xs.min()}, max {xs.max()}"
    lines = [f"# Stage boundaries: {args.task}", "", f"annotation: `{ann_path}`", "",
             "segments: " + " | ".join(f"{i + 1}. {d}" for i, d in enumerate(names)), "",
             f"episodes: {len(rows)}; annotation frames - HDF5 frames: "
             + stat([r["ann_total"] - r["length"] for r in rows]), ""]
    if args.task == "put_back_block":
        lines += ["Raw stage end minus state event (frames):", "",
                  "- stage 1 end - right gripper release (block at centre): "
                  + stat([r["raw_ends"][0] - r["right_release"] for r in rows if r["right_release"] is not None]),
                  "- stage 2 end - left EE lowest point (button press): "
                  + stat([r["raw_ends"][1] - r["left_z_min"] for r in rows]), ""]
    lines += [f"Grid ({GRID} frames) rounding moved ends by: "
              + stat([g - e for r in rows for g, e in zip(r["grid_ends"][:-1], r["raw_ends"][:-1])]), "",
              "| episode | length | raw ends | grid ends | right release | left z min |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['episode']} | {r['length']} | {r['raw_ends']} | {r['grid_ends']} | {r['right_release']} | {r['left_z_min']} |"
              for r in rows]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))

    # ---- frames at the boundaries (first / end of each stage) for a visual check
    if args.frames:
        import cv2
        fdir = out / "check_frames"
        fdir.mkdir(exist_ok=True)
        for ref in refs[: args.frames]:
            st = boundaries[Path(ref.path).name]
            idx = [0] + [s[1] for s in st]
            with EpisodeReader(ref.path) as r:
                frames = r.read_frames(list(cfg.dataset.camera_names)[0], idx)
            tiles = []
            for t, fr in zip(idx, frames):
                fr = cv2.cvtColor(np.ascontiguousarray(fr), cv2.COLOR_RGB2BGR)
                cv2.putText(fr, f"t={t}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                tiles.append(fr)
            cv2.imwrite(str(fdir / f"{Path(ref.path).stem}.jpg"), np.hstack(tiles))
        print(f"boundary frames: {fdir}")
    print(f"\nwritten {out / 'boundaries_gt.json'} ({len(boundaries)} episodes)")


if __name__ == "__main__":
    main()
