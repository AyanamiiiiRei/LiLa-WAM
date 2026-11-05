"""Stage-memory phase 3: do the DiT outputs at the VTT position carry stage structure?

  python analysis/vtt_token_analysis.py [--task swap_blocks]

Input: scripts/step35_vtt_tokens.py output (one folder per run: A, M, S).
Features compared (one vector per chunk):
  VTT token of each run     DiT final-norm output at position 35 (the input VTT is a constant)
  current-frame CLS         the model's visual input (does the token add anything to it?)
  time                      one-hot chunk index (swap_blocks is very regular in time: a token
                            that only matches time does not show stage knowledge)
Stage labels (boundaries_gt.json) are used only to EVALUATE.

  1. linear probe (softmax regression, inputs standardised on the training part):
       train demos -> held-out demos accuracy, and 10-fold leave-demos-out CV accuracy (all 50)
  2. CV accuracy on the "time-ambiguous" chunks, i.e. those the time probe gets wrong
     (near stage ends): only here can a feature show stage information beyond time
  3. the decisive moments: chunk right after move 2 vs right after move 3 (same scene layout,
     different next step): leave-one-demo-out accuracy of a binary probe
  4. unsupervised: k-means (k = 4) on the first 50 principal components -> ARI with the stages
Figures: t-SNE (or PCA if scikit-learn is missing) coloured by stage and by time.
Output: printed tables + artifacts/stage_memory/<task>/vtt_tokens/vtt_token_analysis.md (+ .png)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_STAGES = 4


def load(dirpath):
    eps = {}
    for f in sorted(Path(dirpath).glob("episode_*.npz")):
        z = np.load(f)
        eps[f.stem] = {k: z[k] for k in z.files}
        eps[f.stem]["split"] = str(z["split"])
    return eps


def softmax_probe(Xtr, ytr, Xte, n_cls, l2=1e-3, iters=400, lr=0.05, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (A.shape[1], n_cls))
    b = np.zeros(n_cls)
    Y = np.eye(n_cls)[ytr]
    m1 = np.zeros_like(W); v1 = np.zeros_like(W); m2 = np.zeros_like(b); v2 = np.zeros_like(b)
    for i in range(1, iters + 1):
        Z = A @ W + b
        Z -= Z.max(1, keepdims=True)
        P = np.exp(Z); P /= P.sum(1, keepdims=True)
        G = (P - Y) / len(A)
        gW, gb = A.T @ G + l2 * W, G.sum(0)
        for p_, g_, m_, v_ in ((W, gW, m1, v1), (b, gb, m2, v2)):
            m_[:] = 0.9 * m_ + 0.1 * g_
            v_[:] = 0.999 * v_ + 0.001 * g_ ** 2
            p_ -= lr * (m_ / (1 - 0.9 ** i)) / (np.sqrt(v_ / (1 - 0.999 ** i)) + 1e-8)
    return (B @ W + b).argmax(1)


def ari(a, b):
    from math import comb
    ca, cb = np.unique(a), np.unique(b)
    M = np.array([[np.sum((a == x) & (b == y)) for y in cb] for x in ca])
    s_ij = sum(comb(int(v), 2) for v in M.ravel())
    s_a, s_b = sum(comb(int(v), 2) for v in M.sum(1)), sum(comb(int(v), 2) for v in M.sum(0))
    tot = comb(len(a), 2)
    e = s_a * s_b / tot
    mx = 0.5 * (s_a + s_b)
    return float((s_ij - e) / (mx - e)) if mx != e else 1.0


def kmeans(X, k, seed=0, iters=100, n_init=10):
    rng = np.random.default_rng(seed)
    best, best_lab = np.inf, None
    for _ in range(n_init):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best:
            best, best_lab = inertia, lab
    return best_lab


def pca(X, d):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:d].T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="swap_blocks")
    args = ap.parse_args()
    root = Path(f"artifacts/stage_memory/{args.task}/vtt_tokens")
    runs = {d.name: load(d) for d in sorted(root.iterdir()) if d.is_dir() and any(d.glob("episode_*.npz"))}
    if not runs:
        raise SystemExit(f"no token files under {root}: run scripts/step35_vtt_tokens.py first")
    ref = next(iter(runs.values()))
    names = sorted(ref)
    for r, eps in runs.items():
        if sorted(eps) != names:
            raise SystemExit(f"{r}: different demos than the other runs")
    split = np.concatenate([[ref[n]["split"]] * len(ref[n]["t"]) for n in names])
    demo = np.concatenate([[i] * len(ref[n]["t"]) for i, n in enumerate(names)])
    y = np.concatenate([ref[n]["stage"] for n in names])
    chunk = np.concatenate([ref[n]["t"] // 16 for n in names])
    feats = {f"VTT token, {r}": np.concatenate([eps[n]["tok"] for n in names]) for r, eps in runs.items()}
    feats["current-frame CLS"] = np.concatenate([ref[n]["cls"] for n in names])
    feats["time (chunk index)"] = np.eye(chunk.max() + 1)[chunk].astype(np.float32)

    tr, te = split == "train", split == "heldout"
    folds = np.arange(len(names)) % 10
    rng = np.random.default_rng(0)
    rng.shuffle(folds)
    fold_of = folds[demo]

    def cv_pred(X):
        pred = np.zeros_like(y)
        for f in range(10):
            m = fold_of == f
            pred[m] = softmax_probe(X[~m], y[~m], X[m], N_STAGES)
        return pred

    preds = {k: cv_pred(X) for k, X in feats.items()}
    amb = preds["time (chunk index)"] != y                       # chunks the time probe gets wrong

    # decisive moments: first chunk after move 2 / after move 3 ends (stage 2 start / stage 3 start)
    rows_dec, lab_dec, dem_dec = [], [], []
    off = np.concatenate([[0], np.cumsum([len(ref[n]["t"]) for n in names])[:-1]])
    for i, n in enumerate(names):
        st = ref[n]["stage"]
        for s_, l_ in ((2, 0), (3, 1)):
            idx = np.where(st == s_)[0]
            if len(idx):
                rows_dec.append(off[i] + idx[0]); lab_dec.append(l_); dem_dec.append(i)
    rows_dec, lab_dec, dem_dec = np.array(rows_dec), np.array(lab_dec), np.array(dem_dec)

    lines = [f"# VTT-position token analysis ({args.task}; runs: {', '.join(runs)})", "",
             f"{len(names)} demos ({tr.sum()} train / {te.sum()} held-out chunks); 4 stages; chance by majority "
             f"class {np.bincount(y).max() / len(y):.2f}", "",
             "| feature | probe: train -> held-out | probe: 10-fold CV (all demos) | CV on time-ambiguous chunks "
             f"(n={amb.sum()}) | after move 2 vs after move 3 (LODO, n={len(rows_dec)}) | k-means ARI |",
             "|---|---|---|---|---|---|"]
    for k, X in feats.items():
        acc_te = (softmax_probe(X[tr], y[tr], X[te], N_STAGES) == y[te]).mean()
        acc_cv = (preds[k] == y).mean()
        acc_amb = (preds[k][amb] == y[amb]).mean() if amb.any() else float("nan")
        Xd = X[rows_dec]
        pd_ = np.array([softmax_probe(Xd[dem_dec != d], lab_dec[dem_dec != d], Xd[dem_dec == d], 2)
                        for d in np.unique(dem_dec)], dtype=object)
        acc_dec = np.mean(np.concatenate(pd_) == np.concatenate([lab_dec[dem_dec == d] for d in np.unique(dem_dec)]))
        Z = pca((X - X.mean(0)) / (X.std(0) + 1e-6), min(50, X.shape[1]))
        a = ari(kmeans(Z, N_STAGES), y)
        lines.append(f"| {k} | {acc_te:.2f} | {acc_cv:.2f} | {acc_amb:.2f} | {acc_dec:.2f} | {a:.2f} |")
    text = "\n".join(lines) + "\n"
    (root / "vtt_token_analysis.md").write_text(text)
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from sklearn.manifold import TSNE
            embed, how = (lambda Z: TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)), "t-SNE"
        except Exception:
            embed, how = (lambda Z: pca(Z, 2)), "PCA"
        show = [k for k in feats if not k.startswith("time")]
        fig, axes = plt.subplots(2, len(show), figsize=(4.2 * len(show), 8))
        cols = np.array(["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        for j, k in enumerate(show):
            X = feats[k]
            E = embed(pca((X - X.mean(0)) / (X.std(0) + 1e-6), 50))
            ax = axes[0, j]
            for s_ in range(N_STAGES):
                m = y == s_
                ax.scatter(E[m, 0], E[m, 1], s=4, c=cols[s_], label=["move 1", "move 2", "move 3", "press"][s_])
            ax.set_title(f"{k}\n({how}, colour = stage)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.legend(markerscale=3, fontsize=8)
            ax = axes[1, j]
            sc = ax.scatter(E[:, 0], E[:, 1], s=4, c=chunk, cmap="viridis")
            ax.set_title("colour = chunk index (time)", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=axes[1, :].tolist(), shrink=0.6)
        fig.savefig(root / "vtt_token_analysis.png", dpi=120, bbox_inches="tight")
        print(f"figure: {root / 'vtt_token_analysis.png'}")
    except Exception as ex:
        print(f"(no figure: {ex})")


if __name__ == "__main__":
    main()
