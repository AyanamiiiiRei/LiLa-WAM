"""Stage memory on all tasks (sm_Mall-seed0, time trigger) vs the round-1 baseline.

  python analysis/stage_all_summary.py [runs_dir] [run=sm_Mall-seed0] [eval tag=M-time]

Per task: successes (same eval seeds -> paired), McNemar exact p, mean memory writes per
episode and how many episodes wrote at least one slot. No control run: the difference also
contains the effect of 8,000 extra fine-tuning steps.
"""
import json
import math
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
RUN = sys.argv[2] if len(sys.argv) > 2 else "sm_Mall-seed0"
TAG = sys.argv[3] if len(sys.argv) > 3 else "M-time"
TASKS = ["observe_and_pickup", "put_back_block", "swap_T", "rearrange_blocks", "press_button", "swap_blocks",
         "battery_try", "cover_blocks", "place_block_mat", "blocks_ranking_try"]


def load(run, tag):
    base = R / run / "rmbench_eval"
    dirs = sorted(d for d in base.glob("ckpt_*") if d.is_dir() and d.name.endswith(tag)) if base.is_dir() else []
    if tag == "":
        dirs = [d for d in dirs if "_M-" not in d.name]
    if not dirs or not (dirs[-1] / "episodes.jsonl").exists():
        return None, None
    out = {}
    for line in open(dirs[-1] / "episodes.jsonl"):
        if line.strip():
            e = json.loads(line)
            out[(e["task"], e["seed"])] = e
    return out, dirs[-1]


def mcnemar(X, Y, keys):
    b = sum(X[k]["success"] and not Y[k]["success"] for k in keys)
    c = sum(Y[k]["success"] and not X[k]["success"] for k in keys)
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


B, bdir = load("baseline-seed0", "")
M, mdir = load(RUN, "_" + TAG)
if B is None or M is None:
    raise SystemExit(f"missing results: baseline {bdir}, stage memory {mdir}")
print(f"baseline: {bdir}\nstage memory (time trigger): {mdir}\n")
print("pairs: seeds evaluated in both runs only\n")
print("| task | baseline | stage memory | only M / only baseline, p | writes per episode | episodes with >= 1 write |")
print("|---|---|---|---|---|---|")
for t in TASKS + ["ALL"]:
    ks = [k for k in M if k in B and (t == "ALL" or k[0] == t)]
    if not ks:
        continue
    sb, sm = sum(B[k]["success"] for k in ks), sum(M[k]["success"] for k in ks)
    b, c, p = mcnemar(M, B, ks)
    w = [len(M[k].get("memory_writes") or []) for k in ks]
    print(f"| {t} | {sb}/{len(ks)} | {sm}/{len(ks)} | {b} / {c}, p={p:.2g} | {sum(w) / len(w):.1f} | {sum(x > 0 for x in w)}/{len(w)} |")
