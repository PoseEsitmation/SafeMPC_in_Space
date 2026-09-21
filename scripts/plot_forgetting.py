#!/usr/bin/env python
"""Plot forgetting_matrix.csv files.

    python scripts/plot_forgetting.py --runs runs/cl_s1 --out runs/cl_s1/analysis
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load(root):
    files = sorted(glob.glob(os.path.join(root, "**", "forgetting_matrix.csv"), recursive=True))
    if not files:
        raise SystemExit(f"no forgetting_matrix.csv under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "env" in df and df.env.nunique() > 1:  # keep task families apart
        df["condition"] = df.env + "/" + df.condition
    return df


def matrix(df, cond, filt, value):
    K = int(df.after_task.max()) + 1
    M = np.full((K, K), np.nan)
    sub = df[(df.condition == cond) & (df["filter"] == filt)]
    for (k, j), v in sub.groupby(["after_task", "eval_task"])[value].mean().items():
        M[int(k), int(j)] = v
    return M


def heatmaps(df, value, cmap, fmt, path):
    conds = sorted(df.condition.unique())
    fig, axes = plt.subplots(len(conds), 2, figsize=(10, 4.2 * len(conds)), squeeze=False)
    for r, cond in enumerate(conds):
        Ms = [matrix(df, cond, f, value) for f in ("unfiltered", "filtered")]
        lo, hi = np.nanmin(Ms), np.nanmax(Ms)
        for ax, M, filt in zip(axes[r], Ms, ("unfiltered", "filtered")):
            im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{cond} — {filt} {value}", fontsize=10)
            ax.set(xlabel="evaluated on task j", ylabel="after training task k",
                   xticks=range(M.shape[1]), yticks=range(M.shape[0]))
            for (k, j), v in np.ndenumerate(M):
                if not np.isnan(v):
                    ax.text(j, k, format(v, fmt), ha="center", va="center", fontsize=8,
                            bbox=dict(fc="white", alpha=0.6, lw=0, pad=1))
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def retention(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for color, cond in zip(COLORS, sorted(df.condition.unique())):
        for filt, style in (("unfiltered", "-o"), ("filtered", "--s")):
            s = df[(df.condition == cond) & (df["filter"] == filt) & (df.eval_task == 0)]
            g = s.groupby("after_task").koz_mean.mean()
            ax.plot(g.index, g.values, style, color=color, label=f"{cond} {filt}")
    ax.set(xlabel="after training task k", ylabel="KOZ violations/ep on task 0",
           title="Retention of task 0", xticks=sorted(df.after_task.unique()))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cl_metrics(df, path):
    """ACC = mean_j R[K][j];  BWT = mean_{j<K} R[K][j] - R[j][j]."""
    K = int(df.after_task.max())
    rows = []
    for (cond, seed, filt), s in df.groupby(["condition", "seed", "filter"]):
        R = s.set_index(["after_task", "eval_task"])
        for value in ("reward", "koz_mean"):
            acc = [R.loc[(K, j), value] for j in range(K + 1)]
            bwt = [R.loc[(K, j), value] - R.loc[(j, j), value] for j in range(K)]
            rows.append(dict(condition=cond, seed=seed, filter=filt, metric=value,
                             acc=np.mean(acc), bwt=np.mean(bwt) if bwt else np.nan))
    m = pd.DataFrame(rows).groupby(["condition", "filter", "metric"])[["acc", "bwt"]].mean()
    m.to_csv(path)
    print(m.round(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.runs)
    heatmaps(df, "koz_mean", "Reds", ".1f", os.path.join(args.out, "matrix_koz.png"))
    heatmaps(df, "reward", "viridis", ".0f", os.path.join(args.out, "matrix_reward.png"))
    retention(df, os.path.join(args.out, "retention_koz.png"))
    cl_metrics(df, os.path.join(args.out, "cl_metrics.csv"))


if __name__ == "__main__":
    main()
