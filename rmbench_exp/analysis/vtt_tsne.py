"""t-SNE of per-demo VTTs (step 5 output) + per-task strength/consistency.

  python analysis/vtt_tsne.py artifacts/vtt/rmbench  [--out artifacts/vtt/rmbench/vtt_tsne.png]

Reads <dir>/<task>/episode_diffs.npz (one last-minus-first CLS difference per training demo).
Note: the model is conditioned on the task MEAN (task_cond.npy), a fixed vector per task.
"""
import argparse, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("vtt_dir", type=Path); ap.add_argument("--out", type=Path)
    ap.add_argument("--weak_norm", type=float, default=3.0)
    a = ap.parse_args()
    X, Y = [], []
    for f in sorted(glob.glob(str(a.vtt_dir / "*" / "episode_diffs.npz"))):
        d = np.load(f)["diffs"]; X.append(d); Y += [Path(f).parent.name] * len(d)
    X, Y = np.concatenate(X), np.array(Y); tasks = sorted(set(Y))
    mean = {t: X[Y == t].mean(0) for t in tasks}; norm = {t: float(np.linalg.norm(mean[t])) for t in tasks}
    cons = {t: float(np.mean([a_ @ mean[t] / np.linalg.norm(a_) / norm[t] for a_ in X[Y == t]])) for t in tasks}
    # leave-one-out nearest task-mean (cosine): can a single demo's VTT identify its task?
    loo = {t: 0 for t in tasks}
    for i in range(len(X)):
        best, bs = None, -9
        for t in tasks:
            idx = np.where(Y == t)[0]; idx = idx[idx != i]; c = X[idx].mean(0)
            s = X[i] @ c / np.linalg.norm(X[i]) / np.linalg.norm(c)
            if s > bs: bs, best = s, t
        loo[Y[i]] += best == Y[i]
    for t in tasks:
        print(f"{t:20s} norm {norm[t]:6.2f}  consistency {cons[t]:.2f}  LOO acc {loo[t] / (Y == t).sum():.2f}")
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    Z = TSNE(2, perplexity=30, metric="cosine", init="pca", random_state=0).fit_transform(Xn)
    weak = {t for t in tasks if norm[t] < a.weak_norm}; cm = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    for i, t in enumerate(tasks):
        z = Z[Y == t]; w = t in weak
        ax[0].scatter(z[:, 0], z[:, 1], s=34 if w else 26, color=cm(i), marker="X" if w else "o", alpha=.85,
                      edgecolors="k" if w else "none", linewidths=.4, label=f"{t} (‖VTT‖={norm[t]:.1f})")
        c = z.mean(0); ax[0].text(c[0], c[1], t, fontsize=8, ha="center", weight="bold" if w else "normal",
                                  bbox=dict(fc="white", ec="none", alpha=.6, pad=1))
        ax[1].scatter(norm[t], cons[t], s=90, color=cm(i), marker="X" if w else "o", edgecolors="k", linewidths=.5)
        ax[1].annotate(t, (norm[t], cons[t]), textcoords="offset points", xytext=(6, -3), fontsize=8.5)
    ax[0].set_title(f"t-SNE of per-demo VTT (last-frame CLS − first-frame CLS), {len(X)} demos\n× = weak-VTT tasks", fontsize=11)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3, frameon=False)
    ax[1].set_xlabel("‖task VTT‖"); ax[1].set_ylabel("mean cos(per-demo VTT, task VTT)"); ax[1].grid(alpha=.3)
    ax[1].set_title("Signal strength vs. consistency across demos", fontsize=11)
    plt.tight_layout(); out = a.out or a.vtt_dir / "vtt_tsne.png"; plt.savefig(out, dpi=180); print("written", out)


if __name__ == "__main__":
    main()
