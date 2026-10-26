#!/usr/bin/env python
"""Control experiment A (plan step 10): RoboTwin with only 50 clean demos per task.

Builds a symlink-only copy of a RoboTwin dataset that the MAIN project's unchanged
dataloader can read:
    <dst>/<task>/demo_clean/data/<episode>.hdf5 -> <src>/<task>/demo_clean/data/<episode>.hdf5
(plus instructions/ if present). Original data is never modified.

Then train/evaluate with the main project as usual, using
configs/control_a/robotwin_50clean.yaml (see docs/STEPS.md, step 10A).

Usage:
  python scripts/step10a_make_robotwin_subset.py --src /data/RoboTwin/data-500-50 \
      --dst /data/RoboTwin/control_a_50clean --n 50
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

# The 10 ablation tasks of the LiLa-WAM paper (Sec. 4.2): none of them needs memory,
# and the paper reports their success rates, which gives a reference point.
DEFAULT_TASKS = ["adjust_bottle", "grab_roller", "hanging_mug", "move_stapler_pad", "open_microwave",
                 "press_stapler", "scan_object", "stack_blocks_two", "stamp_seal", "turn_switch"]


def ep_num(p: Path) -> int:
    m = re.findall(r"\d+", p.stem)
    return int(m[-1]) if m else -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="RoboTwin root: <src>/<task>/demo_clean/data/*.hdf5")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if dst == src or src in dst.parents:
        raise SystemExit("--dst must be outside --src")
    for task in args.tasks:
        data = src / task / "demo_clean" / "data"
        eps = sorted(data.glob("*.hdf5"), key=ep_num)[: args.n]
        if len(eps) < args.n:
            raise SystemExit(f"{task}: only {len(eps)} clean episodes in {data}")
        out = dst / task / "demo_clean" / "data"
        out.mkdir(parents=True, exist_ok=True)
        for e in eps:
            link = out / e.name
            if not link.exists():
                os.symlink(e, link)
        instr = src / task / "demo_clean" / "instructions"
        if instr.is_dir() and not (dst / task / "demo_clean" / "instructions").exists():
            os.symlink(instr, dst / task / "demo_clean" / "instructions")
        print(f"{task}: {len(eps)} episodes linked")
    print(f"\nDone: {dst}")


if __name__ == "__main__":
    main()
