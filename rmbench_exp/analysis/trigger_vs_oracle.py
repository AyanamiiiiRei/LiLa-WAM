"""Stage-memory phase 2: closed-loop comparison of memory-write triggers on the M checkpoint.

  python analysis/trigger_vs_oracle.py [runs_dir]

Groups (same checkpoint, same 50 seeds): M (oracle), M-proprio, M-head, M-time, plus A and
M-empty as references. For each group: success with Wilson 95% CI, McNemar exact p vs M
(oracle) and vs A, and the share of M's gain over A that is kept:
    kept = (succ_group - succ_A) / (succ_M - succ_A)
Write timing (non-oracle triggers; the oracle still ran and is logged per chunk): for each
write k, chunks between the trigger's write and the chunk where the oracle first reported
stage k done. "false" = a write for a stage the oracle never reported in that episode.
"""
import json
import math
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
G = {"A": "sm_A-seed0/rmbench_eval/ckpt_8000", "M-empty": "sm_M-seed0/rmbench_eval/ckpt_8000_M-empty",
     "M (oracle)": "sm_M-seed0/rmbench_eval/ckpt_8000", "M-proprio": "sm_M-seed0/rmbench_eval/ckpt_8000_M-proprio",
     "M-head": "sm_M-seed0/rmbench_eval/ckpt_8000_M-head", "M-time": "sm_M-seed0/rmbench_eval/ckpt_8000_M-time",
     "S (learned switch)": "sm_S-seed0/rmbench_eval/ckpt_1500"}


def load(p):
    f = R / p / "episodes.jsonl"
    if not f.is_file():
        return None
    E = {}
    for line in open(f):
        if line.strip():
            e = json.loads(line)
            E[e["seed"]] = e
    return E


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def mcnemar(X, Y):
    s = sorted(set(X) & set(Y))
    b = sum(X[i]["success"] and not Y[i]["success"] for i in s)
    c = sum(Y[i]["success"] and not X[i]["success"] for i in s)
    n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


data = {g: load(p) for g, p in G.items()}
ref, base = data["M (oracle)"], data["A"]
sm = sum(e["success"] for e in ref.values()) if ref else None
sa = sum(e["success"] for e in base.values()) if base else None
print("| group | success | 95% CI | vs M (only this / only M, p) | vs A (only this / only A, p) | kept share of M's gain |")
print("|---|---|---|---|---|---|")
for g, E in data.items():
    if E is None:
        print(f"| {g} | (no results) | | | | |")
        continue
    k, n = sum(e["success"] for e in E.values()), len(E)
    lo, hi = wilson(k, n)
    vm = "%d / %d, p=%.2g" % mcnemar(E, ref) if ref and g != "M (oracle)" else "-"
    va = "%d / %d, p=%.2g" % mcnemar(E, base) if base and g != "A" else "-"
    kept = f"{(k - sa) / (sm - sa):.2f}" if sm is not None and sa is not None and sm != sa and g not in ("A",) else "-"
    print(f"| {g} | {k}/{n} | {lo:.0%}-{hi:.0%} | {vm} | {va} | {kept} |")

print("\nwrite timing vs the oracle (chunks; negative = trigger wrote before the oracle)")
for g in ("M-proprio", "M-head", "M-time", "S (learned switch)"):
    E = data[g]
    if E is None:
        continue
    per = {0: [], 1: [], 2: []}
    false = {0: 0, 1: 0, 2: 0}
    for e in E.values():
        tr = e.get("oracle_trace") or []
        first = {}
        for c, v in enumerate(tr):
            for kk in range(v or 0):
                first.setdefault(kk, c)
        for c, slot in e.get("memory_writes") or []:
            if slot in first:
                per[slot].append(c - first[slot])
            else:
                false[slot] += 1
    for slot in (0, 1, 2):
        v = sorted(per[slot])
        if v or false[slot]:
            med = v[len(v) // 2] if v else None
            within = sum(abs(x) <= 1 for x in v)
            print(f"  {g} write {slot + 1}: matched {len(v)} (median {med}, within +-1: {within}, "
                  f"min {min(v) if v else '-'}, max {max(v) if v else '-'}), false writes {false[slot]}")
