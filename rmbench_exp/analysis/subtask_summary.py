"""Sub-task design: closed-loop results of plan 1, plan 2, the control and the round-1 baseline.

  python analysis/subtask_summary.py [runs_dir]

Per task: successes out of the episodes run (same seeds for every run), McNemar exact p of
P1 and P2 vs the round-1 baseline (the warm start; P1/P2 were fine-tuned 20,000 more steps, no
matched control was run), and for P1/P2 the mean
number of sub-task switches per episode (from the episode log). Pooled over tasks at the end.
Also pastes the chooser reports (held-out demos) if present.
"""
import json
import math
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
RUNS = {"baseline (round 1)": "baseline-seed0", "P1 (joint)": "sub_P1-seed0", "P2 (separate)": "sub_P2-seed0"}
TASKS = ["observe_and_pickup", "put_back_block", "swap_T", "rearrange_blocks", "press_button", "swap_blocks",
         "battery_try", "cover_blocks", "place_block_mat", "blocks_ranking_try"]


def load(run):
    base = R / run / "rmbench_eval"
    if not base.is_dir():
        return None
    dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("ckpt_") and "_M-" not in d.name)
    if not dirs:
        return None
    out = {}
    for line in open(dirs[-1] / "episodes.jsonl") if (dirs[-1] / "episodes.jsonl").exists() else []:
        if line.strip():
            e = json.loads(line)
            out[(e.get("task"), e["seed"])] = e
    return out


def mcnemar(X, Y, keys):
    b = sum(X[k]["success"] and not Y[k]["success"] for k in keys)
    c = sum(Y[k]["success"] and not X[k]["success"] for k in keys)
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


data = {name: load(run) for name, run in RUNS.items()}
head = "| task | " + " | ".join(RUNS) + " | P1 vs baseline (only P1 / only base, p) | P2 vs baseline | switches/ep P1, P2 |"
print(head)
print("|---" * (len(RUNS) + 4) + "|")
pooled = {n: [0, 0] for n in RUNS}
for t in TASKS + ["ALL"]:
    cells = []
    for name, d in data.items():
        if d is None:
            cells.append("-")
            continue
        ks = [k for k in d if t == "ALL" or k[0] == t]
        s = sum(d[k]["success"] for k in ks)
        cells.append(f"{s}/{len(ks)}" if ks else "-")
    cmp = []
    C = data["baseline (round 1)"]
    for name in ("P1 (joint)", "P2 (separate)"):
        P = data[name]
        if C is None or P is None:
            cmp.append("-")
            continue
        ks = [k for k in P if k in C and (t == "ALL" or k[0] == t)]
        cmp.append("%d / %d, p=%.2g" % mcnemar(P, C, ks) if ks else "-")
    sw = []
    for name in ("P1 (joint)", "P2 (separate)"):
        P = data[name]
        ks = [k for k in (P or {}) if t == "ALL" or k[0] == t]
        sw.append(f"{sum(len(P[k].get('subtask_log') or []) for k in ks) / len(ks):.1f}" if ks else "-")
    print(f"| {t} | " + " | ".join(cells) + f" | {cmp[0]} | {cmp[1]} | {', '.join(sw)} |")

for run in ("sub_P1-seed0", "sub_P2-seed0"):
    f = R / run / "chooser_report.md"
    if f.exists():
        print("\n" + f.read_text())
