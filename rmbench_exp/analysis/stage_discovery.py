"""Stage-memory phase 3 (diagnostic C): can the stages be found without boundary labels?

  python analysis/stage_discovery.py [--surprise artifacts/stage_memory/swap_blocks/surprise/sm_A-seed0]

Every method places exactly 3 boundaries per demo (the number of written stages is given,
as it is to M), on the 16-frame chunk grid, at least 2 chunks from the episode start/end and
at least 4 chunks apart. Truth: boundaries_gt.json (move ends; equal to the oracle, step 26).

  no labels used
    surprise peaks        3 highest foresight-surprise chunks of the memoryless model (step 31)
    CLS segments          contiguous 4-segment split minimising within-segment variance of the
                          (L2-normalised) current-frame CLS          [plan baseline 1]
    uniform progress      length x 1/4, 2/4, 3/4                      [plan baseline 2]
    foresight-change seg. same split on the foresight local change   [plan baseline 3]
    CLS-change direction  the phase-2 offline segmentation (unit CLS(t+32) - CLS(t))
    random                3 random valid positions (mean of 200 draws)
  labels used to set ONE number on the 45 training demos (reported as such)
    surprise peaks, shift the peak-to-boundary offset (chunks) that fits the training demos best
    mean time             the training demos' mean boundary times (= phase-2 time trigger)

Metrics (train / held-out / all): boundary hits within +-1 chunk (one-to-one), ARI of the
per-chunk stage labels, mean |error| in chunks. Also: surprise around the true boundaries
(z-scored per demo, averaged), to see whether surprise peaks at, before or after them.
Output: printed tables + <surprise dir>/stage_discovery.md (+ .png if matplotlib is available).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GRID, K, EDGE, GAP = 16, 3, 2, 4


# ------------------------------------------------------------------ segmentation tools
def contiguous_split(X, k_cuts, edge=EDGE, gap=GAP):
    """Cut indices (chunk positions) of a contiguous split of rows of X into k_cuts+1 segments
    minimising the total within-segment sum of squares; constrained like every other method."""
    n = len(X)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
    cs2 = np.concatenate([[0.0], np.cumsum((X ** 2).sum(1))])

    def sse(i, j):
        m = j - i
        s = cs[j] - cs[i]
        return cs2[j] - cs2[i] - (s @ s) / m

    INF = np.inf
    F = np.full((k_cuts + 2, n + 1), INF)
    P = np.zeros((k_cuts + 2, n + 1), int)
    F[0, 0] = 0.0
    for seg in range(1, k_cuts + 2):
        for j in range(1, n + 1):
            for i in range(0, j):
                if F[seg - 1, i] == INF:
                    continue
                if seg > 1 and (i < edge or i > n - edge):            # i is a cut position
                    continue
                if seg > 1 and j < n and j - i < gap:
                    continue
                if seg == 1 and i != 0:
                    continue
                v = F[seg - 1, i] + sse(i, j)
                if v < F[seg, j]:
                    F[seg, j], P[seg, j] = v, i
    cuts, j = [], n
    for seg in range(k_cuts + 1, 1, -1):
        i = P[seg, j]
        cuts.append(i)
        j = i
    return sorted(cuts)


def top_peaks(score, k_cuts, edge=EDGE, gap=GAP):
    """k_cuts positions maximising the summed score, pairwise >= gap apart, edge from the ends."""
    n = len(score)
    valid = [i for i in range(edge, n - edge + 1) if i < n]
    best, best_set = -np.inf, None
    # small n (~40): exhaustive over ordered triples with pruning
    for a in valid:
        for b in valid:
            if b - a < gap:
                continue
            for c in valid:
                if c - b < gap:
                    continue
                v = score[a] + score[b] + score[c]
                if v > best:
                    best, best_set = v, [a, b, c]
    return best_set


def labels_from_cuts(cuts, n):
    lab = np.zeros(n, int)
    for c in cuts:
        lab[min(max(c, 0), n):] += 1
    return lab


def ari(a, b):
    from math import comb
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == x) & (b == y)) for y in cb] for x in ca])
    s_ij = sum(comb(int(v), 2) for v in M.ravel())
    s_a = sum(comb(int(v), 2) for v in M.sum(1))
    s_b = sum(comb(int(v), 2) for v in M.sum(0))
    tot = comb(len(a), 2)
    exp = s_a * s_b / tot if tot else 0
    mx = 0.5 * (s_a + s_b)
    return float((s_ij - exp) / (mx - exp)) if mx != exp else 1.0


def hits(pred, true, tol=1):
    used, h = set(), 0
    for t in true:
        cand = [(abs(p - t), i) for i, p in enumerate(pred) if i not in used and abs(p - t) <= tol]
        if cand:
            used.add(min(cand)[1])
            h += 1
    return h


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="swap_blocks")
    ap.add_argument("--surprise", default=None)
    args = ap.parse_args()
    art = Path(f"artifacts/stage_memory/{args.task}")
    sdir = Path(args.surprise or art / "surprise" / "sm_A-seed0")
    bounds = json.loads((art / "boundaries_gt.json").read_text())

    eps = {}
    for f in sorted(sdir.glob("episode_*.npz")):
        z = np.load(f)
        name = f.stem + ".hdf5"
        cls = np.load(art / "cls" / (f.stem + ".npy")).astype(np.float32)
        t = z["t"]
        n = len(t)
        true = [bounds[name][k][1] // GRID for k in range(K)]          # chunk index of each true end
        c = cls[t]
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        d = np.array([cls[min(x + 32, len(cls) - 1)] - cls[x] for x in t])
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
        eps[name] = dict(n=n, true=true, err=z["err"], delta=z["delta"], cls=c, dcls=d,
                         split=str(z["split"]), length=int(z["length"]))
    if not eps:
        raise SystemExit(f"no surprise files in {sdir}: run scripts/step31_surprise.py first")
    train = [e for e in eps.values() if e["split"] == "train"]

    # label-tuned numbers (training demos only)
    def peak_cuts(e, shift):
        return [min(max(p + shift, 0), e["n"] - 1) for p in top_peaks(e["err"], K)]
    best_shift = max(range(-3, 4), key=lambda s: sum(hits(peak_cuts(e, s), e["true"]) for e in train))
    mean_ends = np.mean([[x for x in e["true"]] for e in train], axis=0)

    rng = np.random.default_rng(0)
    methods = {
        "surprise peaks": lambda e: top_peaks(e["err"], K),
        f"surprise peaks, shift {best_shift:+d} (tuned)": lambda e: peak_cuts(e, best_shift),
        "CLS segments": lambda e: contiguous_split(e["cls"], K),
        "foresight-change segments": lambda e: contiguous_split(e["delta"] / (np.linalg.norm(
            e["delta"], axis=1, keepdims=True) + 1e-8), K),
        "CLS-change direction": lambda e: contiguous_split(e["dcls"], K),
        "uniform progress": lambda e: [int(np.ceil(e["length"] * q / 4 / GRID)) for q in (1, 2, 3)],
        "mean time (tuned)": lambda e: [int(round(x)) for x in mean_ends],
    }

    def random_cuts(e):
        valid = list(range(EDGE, e["n"] - EDGE + 1))
        while True:
            s = sorted(rng.choice(valid, K, replace=False))
            if all(b - a >= GAP for a, b in zip(s, s[1:])):
                return s

    rows = []
    for name, fn in list(methods.items()) + [("random", None)]:
        res = {}
        for split in ("train", "heldout", "all"):
            es = [e for e in eps.values() if split == "all" or e["split"] == split]
            h = a = err = 0.0
            for e in es:
                truth = labels_from_cuts(e["true"], e["n"])
                if fn is None:
                    draws = [random_cuts(e) for _ in range(200)]
                    h += np.mean([hits(c, e["true"]) for c in draws])
                    a += np.mean([ari(labels_from_cuts(c, e["n"]), truth) for c in draws])
                    err += np.mean([np.mean(np.abs(np.array(c) - e["true"])) for c in draws])
                else:
                    c = sorted(fn(e))
                    h += hits(c, e["true"])
                    a += ari(labels_from_cuts(c, e["n"]), truth)
                    err += np.mean(np.abs(np.array(c) - np.array(e["true"])))
            res[split] = (h, K * len(es), a / len(es), err / len(es))
        rows.append((name, res))

    lines = [f"# Stage discovery without labels ({args.task}; surprise from {sdir.name})", "",
             f"{len(train)} train / {len(eps) - len(train)} held-out demos; 3 boundaries per demo; "
             "hits = within +-1 chunk", "",
             "| method | hits train | hits held-out | ARI train | ARI held-out | mean abs error (chunks), all |",
             "|---|---|---|---|---|---|"]
    for name, r in rows:
        lines.append(f"| {name} | {r['train'][0]:.0f}/{r['train'][1]} | {r['heldout'][0]:.0f}/{r['heldout'][1]} | "
                     f"{r['train'][2]:.2f} | {r['heldout'][2]:.2f} | {r['all'][3]:.1f} |")

    # surprise around the true boundaries
    win = range(-6, 7)
    acc = {o: [] for o in win}
    for e in eps.values():
        z = (e["err"] - e["err"].mean()) / (e["err"].std() + 1e-8)
        for b in e["true"]:
            for o in win:
                if 0 <= b + o < e["n"]:
                    acc[o].append(z[b + o])
    lines += ["", "surprise (z-score within each demo) around the true boundaries, all demos:", "",
              "| chunks from boundary | " + " | ".join(f"{o:+d}" for o in win) + " |",
              "|---|" + "---|" * len(win),
              "| mean z | " + " | ".join(f"{np.mean(acc[o]):+.2f}" for o in win) + " |"]
    text = "\n".join(lines) + "\n"
    (sdir / "stage_discovery.md").write_text(text)
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False)
        for ax, (name, e) in zip(axes, list(eps.items())[:3]):
            ax.plot(np.arange(e["n"]), e["err"], color="#4060a0")
            for b in e["true"]:
                ax.axvline(b, color="#c04040", ls="--", lw=1)
            for p in top_peaks(e["err"], K):
                ax.plot(p, e["err"][p], "o", color="#202020")
            ax.set_title(f"{name}: foresight surprise per chunk (red = true move ends, dots = 3 peaks)", fontsize=9)
            ax.set_ylabel("1 - cos")
        axes[-1].set_xlabel("chunk index (16 frames)")
        fig.tight_layout()
        fig.savefig(sdir / "stage_discovery.png", dpi=120)
        print(f"figure: {sdir / 'stage_discovery.png'}")
    except Exception as ex:                               # plotting is optional
        print(f"(no figure: {ex})")


if __name__ == "__main__":
    main()
