#!/usr/bin/env python
"""Plan step 11.2: classify every failed rollout by watching its video.

  init       creates <eval_dir>/failure_labels.csv with one row per failed episode
             (task, seed, steps) and an empty `label` column to fill in
  summarize  counts labels per task and per M(1)/M(n) group

Labels (edit LABELS below to extend; keep them short):
  memory        manipulation itself was fine, but the wrong object/order/count
                (repeats a done step, forgets a hidden object, wrong number of presses)
  manipulation  grasp failed, imprecise placement, knocked something over
  stuck         timed out / froze / oscillated without progress
  other         anything else (explain in `note`)

Videos: RMBench writes them to eval_result/<task>/<policy>/<task_config>/<ckpt>/<time>/episode<N>.mp4
(N = order of the evaluated episodes, matching the order of lines in episodes.jsonl).

Usage:
  python analysis/failure_labels.py init runs/baseline/<ts>/rmbench_eval/ckpt_30000
  python analysis/failure_labels.py summarize runs/baseline/<ts>/rmbench_eval/ckpt_30000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rmbench_lila.tasks import task_kind  # noqa: E402

LABELS = ["memory", "manipulation", "stuck", "other"]
FIELDS = ["task", "kind", "episode_index_in_task", "seed", "env_steps", "label", "note"]


def cmd_init(d: Path):
    out = d / "failure_labels.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting your labels")
    rows, idx = [], defaultdict(int)
    for line in (d / "episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        i = idx[r["task"]]
        idx[r["task"]] += 1
        if not r["success"]:
            rows.append({"task": r["task"], "kind": task_kind(r["task"]), "episode_index_in_task": i,
                         "seed": r["seed"], "env_steps": r["env_steps"], "label": "", "note": ""})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} failures -> {out}\nFill `label` with one of {LABELS}.")


def cmd_summarize(d: Path):
    rows = list(csv.DictReader(open(d / "failure_labels.csv")))
    bad = [r for r in rows if r["label"] not in LABELS]
    if bad:
        print(f"warning: {len(bad)} rows unlabeled or with unknown labels (e.g. {bad[0]['task']} seed {bad[0]['seed']})")
    per_task = defaultdict(Counter)
    per_kind = defaultdict(Counter)
    for r in rows:
        if r["label"] in LABELS:
            per_task[r["task"]][r["label"]] += 1
            per_kind[r["kind"]][r["label"]] += 1
    L = ["| task | type | " + " | ".join(LABELS) + " | memory share |", "|---|---|" + "---|" * (len(LABELS) + 1)]
    for t in sorted(per_task, key=lambda t: (task_kind(t), t)):
        c = per_task[t]
        n = sum(c.values())
        L.append(f"| {t} | {task_kind(t)} | " + " | ".join(str(c[l]) for l in LABELS) + f" | {c['memory'] / n:.0%} |")
    for k in sorted(per_kind):
        c = per_kind[k]
        n = sum(c.values())
        L.append(f"| *all {k}* | {k} | " + " | ".join(str(c[l]) for l in LABELS) + f" | {c['memory'] / n:.0%} |")
    (d / "failure_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["init", "summarize"])
    ap.add_argument("eval_dir")
    a = ap.parse_args()
    {"init": cmd_init, "summarize": cmd_summarize}[a.mode](Path(a.eval_dir))


if __name__ == "__main__":
    main()
