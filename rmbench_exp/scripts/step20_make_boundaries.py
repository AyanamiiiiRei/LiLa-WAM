#!/usr/bin/env python
"""Stage-memory experiment, step 1: stage boundaries from RMBench's language annotations.

  python scripts/step20_make_boundaries.py --task swap_blocks
  -> artifacts/stage_memory/<task>/boundaries_gt.json  (+ report.md, check frames)

Boundary source (--source, default per task):
  events (swap_blocks): from the robot state, the SAME definition the eval oracle uses:
      move k ends when, after the k-th gripper release, both arms are back at their start
      pose. (The annotation text is regenerated during a move - "left block" becomes
      "right block" once it has moved - so merging identical descriptions does not give
      the 4 stages.) The report compares these ends with the annotation segment ends.
  annotation (put_back_block): merge consecutive identical annotation segments.
Every stage end is rounded UP to the 16-frame grid; the last stage runs to the episode end.
Writes frames at each boundary for a visual check (--frames N episodes).
"""

from __future__ import annotations

import json
from pathlib import Path

import _common
import numpy as np

from rmbench_lila.episodes import EpisodeReader, scan_episodes
from rmbench_lila.stages import GRID, grid_stages, merge_annotation, raw_ends, swap_blocks_event_ends

DEFAULT_SOURCE = {"swap_blocks": "events", "put_back_block": "annotation"}


def state_events(reader: EpisodeReader) -> dict:
    import h5py  # noqa: F401
    f = reader.f
    rg = f["state/right_ee_joint_states"][()].reshape(-1)
    lz = f["state/left_ee_poses"][()][:, 2]
    rel = [t for t in range(1, len(rg)) if rg[t] > 0.5 >= rg[t - 1]]
    return {"right_release": rel[0] if rel else None, "left_z_min": int(np.argmin(lz))}


def main():
    p = _common.parser(__doc__)
    p.add_argument("--task", default="swap_blocks")
    p.add_argument("--source", choices=["events", "annotation"], default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--frames", type=int, default=5, help="episodes to dump boundary frames for")
    args = p.parse_args()
    cfg = _common.load(args)
    source = args.source or DEFAULT_SOURCE.get(args.task, "annotation")
    if source == "events" and args.task != "swap_blocks":
        raise SystemExit("event boundaries are only defined for swap_blocks")
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
        ann_ends = raw_ends(merged)
        with EpisodeReader(ref.path) as r:
            n = r.num_frames(list(cfg.dataset.camera_names)[0])
            ev = state_events(r)
            if source == "events":
                f = r.f
                ends = swap_blocks_event_ends(
                    f["state/left_ee_poses"][()][:, :3], f["state/right_ee_poses"][()][:, :3],
                    f["state/left_ee_joint_states"][()].reshape(-1), f["state/right_ee_joint_states"][()].reshape(-1))
                ends = ends + [n - 1]
            else:
                if names is None:
                    names = [d for d, _ in merged]
                if len(merged) != len(names):
                    raise ValueError(f"{key}: {len(merged)} segments, expected {len(names)}: {merged}")
                ends = ann_ends
        stages = grid_stages(ends, n)
        boundaries[Path(ref.path).name] = stages
        # distance from each (non-final) end to the nearest annotation segment end
        ann_gap = [min(abs(e - a) for a in ann_ends) for e in ends[:-1]]
        rows.append({"episode": Path(ref.path).name, "length": n, "raw_ends": ends,
                     "grid_ends": [s[1] for s in stages], **ev, "ann_gap": ann_gap,
                     "ann_total": int(sum(k for _, k in merged))})

    (out / "boundaries_gt.json").write_text(json.dumps(boundaries, indent=1))

    # ---- report
    def stat(xs):
        xs = np.asarray([x for x in xs if x is not None])
        return f"median {np.median(xs):.0f}, min {xs.min()}, max {xs.max()}"
    lines = [f"# Stage boundaries: {args.task} (source: {source})", "", f"annotation: `{ann_path}`", "",
             ("segments: " + " | ".join(f"{i + 1}. {d}" for i, d in enumerate(names))) if names else
             "stages: move 1 | move 2 | move 3 | press the button (only moves 1-3 are written to memory)", "",
             "stage end - nearest annotation segment end (frames): "
             + stat([abs(g) for r in rows for g in r["ann_gap"]]), "",
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
              "| episode | length | raw ends | grid ends | gap to annotation |", "|---|---|---|---|---|"]
    lines += [f"| {r['episode']} | {r['length']} | {r['raw_ends']} | {r['grid_ends']} | {r['ann_gap']} |"
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
