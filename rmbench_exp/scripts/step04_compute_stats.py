#!/usr/bin/env python
"""Plan step 4: normalization statistics on RMBench. NEVER deletes or modifies data.

Episodes with suspicious values are listed in the output ("flagged") and, by
default, still included. Use --exclude_flagged only after looking at them.

Usage:
  python scripts/step04_compute_stats.py --config configs/exp/baseline.yaml
Output: paths.norm_stats from the config (default artifacts/stats/stats-rmbench.json)
"""

from __future__ import annotations

from pathlib import Path

import _common

from rmbench_lila.episodes import select_episodes
from rmbench_lila.schema import get_action_layout, get_state_layout
from rmbench_lila.stats import compute_stats, save_stats


def main():
    p = _common.parser(__doc__)
    p.add_argument("--split", default="train", choices=["train", "all"],
                   help="train = exclude the held-out episodes used for open-loop eval")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--exclude_flagged", action="store_true")
    p.add_argument("--out", default=None, help="defaults to paths.norm_stats")
    args = p.parse_args()
    cfg = _common.load(args)

    refs = select_episodes(cfg, args.split)
    stats = compute_stats(refs, get_state_layout(cfg.rmbench.state_layout),
                          get_action_layout(cfg.rmbench.action_layout),
                          cfg.rmbench.action_alignment, args.workers, args.exclude_flagged,
                          state_transforms=cfg.rmbench.get("state_transforms", []))
    out = Path(args.out or cfg.paths.norm_stats)
    save_stats(stats, out)

    meta = stats["robotwin2"]["meta"]
    print(f"\nSaved {out}")
    print(f"files: {meta['total_files']}  used: {meta['valid_files']}  flagged: {meta['flagged_files']}")
    for row in stats["flagged"][:20]:
        print(f"  FLAGGED {row['task']}: {Path(row['path']).name}: {'; '.join(row['issues'])}")
    pt = stats["robotwin2"]["per_task"]
    print(f"state transforms: {meta['state_transforms'] or 'none'}")
    for task, d in pt.items():
        if d["quat_flips_raw"] or d["quat_flips_used"]:
            print(f"  quaternion sign flips {task}: raw {d['quat_flips_raw']} -> used {d['quat_flips_used']}")
    a = stats["robotwin2"]["action"]
    print("action min:", [round(x, 3) for x in a["min"]])
    print("action max:", [round(x, 3) for x in a["max"]])


if __name__ == "__main__":
    main()
