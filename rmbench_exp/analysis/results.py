#!/usr/bin/env python
"""Plan step 11.1 / 11.3: success rates with 95% CIs, grouped by M(1)/M(n), and paired
comparisons between runs.

Reads the per-episode logs written by the LiLa_WAM adapter:
    <run>/rmbench_eval/ckpt_<step>/episodes.jsonl

Usage:
  python analysis/results.py table runs/baseline/<ts>
  python analysis/results.py compare runs/baseline/<ts> runs/state_hist8/<ts>
  (add --ckpt <step> when a run was evaluated with several checkpoints)
Writes a markdown table next to the logs (or to --out).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rmbench_lila.evalstats import mcnemar_exact, wilson_ci  # noqa: E402
from rmbench_lila.tasks import task_kind  # noqa: E402


def load_episodes(run: Path, ckpt: str | None) -> tuple[Path, dict[str, dict]]:
    base = run / "rmbench_eval"
    dirs = sorted(base.glob(f"ckpt_{ckpt}" if ckpt else "ckpt_*"))
    if not dirs:
        raise SystemExit(f"no evaluations under {base}")
    if len(dirs) > 1 and not ckpt:
        raise SystemExit(f"several checkpoints evaluated in {run}: {[d.name for d in dirs]}; pass --ckpt")
    by_task: dict[str, dict] = defaultdict(dict)   # task -> {seed: success}
    dup = 0
    for line in (dirs[0] / "episodes.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["seed"] in by_task[r["task"]]:
            dup += 1
        by_task[r["task"]][r["seed"]] = r["success"]   # last result wins for a re-run seed
    if dup:
        print(f"warning: {dup} duplicate (task, seed) entries in {dirs[0]} (re-runs?); kept the last")
    return dirs[0], by_task


def table(by_task: dict) -> list[str]:
    L = ["| task | type | n | success | rate | 95% CI |", "|---|---|---|---|---|---|"]
    groups = defaultdict(lambda: [0, 0])
    for task in sorted(by_task, key=lambda t: (task_kind(t), t)):
        res = list(by_task[task].values())
        k, n = sum(res), len(res)
        lo, hi = wilson_ci(k, n)
        L.append(f"| {task} | {task_kind(task)} | {n} | {k} | {k / n:.1%} | [{lo:.1%}, {hi:.1%}] |")
        groups[task_kind(task)][0] += k
        groups[task_kind(task)][1] += n
    rates = [sum(v.values()) / len(v) for v in by_task.values()]
    L.append(f"| **mean over tasks** | | | | **{sum(rates) / len(rates):.1%}** | |")
    for g, (k, n) in sorted(groups.items()):
        lo, hi = wilson_ci(k, n)
        L.append(f"| *all {g} episodes* | {g} | {n} | {k} | {k / n:.1%} | [{lo:.1%}, {hi:.1%}] |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["table", "compare"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", nargs="*", default=None, help="checkpoint step per run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ck = args.ckpt or [None] * len(args.runs)

    if args.mode == "table":
        d, by_task = load_episodes(Path(args.runs[0]), ck[0])
        L = [f"# Results: {args.runs[0]} ({d.name})", ""] + table(by_task)
        out = Path(args.out) if args.out else d / "results.md"
    else:
        if len(args.runs) != 2:
            raise SystemExit("compare needs exactly two runs: A B")
        (da, a), (db, b) = (load_episodes(Path(r), c) for r, c in zip(args.runs, ck))
        L = [f"# Paired comparison", "", f"A = {args.runs[0]} ({da.name})", f"B = {args.runs[1]} ({db.name})", "",
             "Paired on identical eval seeds; p = exact McNemar test.", "",
             "| task | type | paired n | A | B | B − A | only A ok | only B ok | p |", "|---|---|---|---|---|---|---|---|---|"]
        for task in sorted(set(a) & set(b), key=lambda t: (task_kind(t), t)):
            m = mcnemar_exact(a[task], b[task])
            L.append(f"| {task} | {task_kind(task)} | {m['paired_n']} | {m['rate_a']:.1%} | {m['rate_b']:.1%} | "
                     f"{m['rate_b'] - m['rate_a']:+.1%} | {m['a_only']} | {m['b_only']} | {m['p_value']:.3f} |")
        only = sorted(set(a) ^ set(b))
        if only:
            L += ["", f"Tasks evaluated in only one run (not compared): {only}"]
        out = Path(args.out) if args.out else Path("comparison.md")
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
