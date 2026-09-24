#!/usr/bin/env python
"""Plot forgetting_matrix.csv files (averaged over seeds).

    python scripts/plot_forgetting.py --runs runs/cl_s2 --out runs/cl_s2/analysis
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
METRICS = [("koz_mean", "Reds", ".1f"), ("att_err_mean_deg", "Purples", ".0f"), ("reward", "viridis", ".0f")]


def load(root):
    files = sorted(glob.glob(os.path.join(root, "**", "forgetting_matrix.csv"), recursive=True))
    if not files:
        raise SystemExit(f"no forgetting_matrix.csv under {root}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "env" in df and df.env.nunique() > 1:  # keep task families apart
        df["condition"] = df.env + "/" + df.condition
    print(df.groupby("condition").seed.unique().to_string())
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
        n = df[df.condition == cond].seed.nunique()
        for ax, M, filt in zip(axes[r], Ms, ("unfiltered", "filtered")):
            im = ax.imshow(M, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{cond} — {filt} {value} (mean of {n} seeds)", fontsize=10)
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


def retention(df, metrics, path):
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 4.5), squeeze=False)
    for ax, value in zip(axes[0], metrics):
        for color, cond in zip(COLORS, sorted(df.condition.unique())):
            for filt, style in (("unfiltered", "-o"), ("filtered", "--s")):
                s = df[(df.condition == cond) & (df["filter"] == filt) & (df.eval_task == 0)]
                g = s.groupby("after_task")[value]
                ax.errorbar(g.mean().index, g.mean().values, yerr=g.std().fillna(0).values,
                            fmt=style, color=color, capsize=3, label=f"{cond} {filt}")
        ax.set(xlabel="after training task k", ylabel=f"{value} on task 0",
               title=f"Retention of task 0 — {value} (mean ± sd over seeds)",
               xticks=sorted(df.after_task.unique()))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def cl_metrics(df, metrics, path):
    """ACC = mean_j R[K][j];  BWT = mean_{j<K} R[K][j] - R[j][j]; mean and sd over seeds."""
    K = int(df.after_task.max())
    rows = []
    for (cond, seed, filt), s in df.groupby(["condition", "seed", "filter"]):
        R = s.set_index(["after_task", "eval_task"])
        for value in metrics:
            rows.append(dict(condition=cond, seed=seed, filter=filt, metric=value,
                             acc=np.mean([R.loc[(K, j), value] for j in range(K + 1)]),
                             bwt=np.mean([R.loc[(K, j), value] - R.loc[(j, j), value]
                                          for j in range(K)])))
    m = (pd.DataFrame(rows).groupby(["condition", "filter", "metric"])[["acc", "bwt"]]
         .agg(["mean", "std", "count"]))
    m.to_csv(path)
    print(m.round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = load(args.runs)
    metrics = [m for m, _, _ in METRICS if m in df]
    for value, cmap, fmt in METRICS:
        if value in df:
            heatmaps(df, value, cmap, fmt, os.path.join(args.out, f"matrix_{value}.png"))
    retention(df, [m for m in ("koz_mean", "att_err_mean_deg") if m in df],
              os.path.join(args.out, "retention_task0.png"))
    cl_metrics(df, metrics, os.path.join(args.out, "cl_metrics.csv"))


if __name__ == "__main__":
    main()
