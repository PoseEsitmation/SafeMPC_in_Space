#!/usr/bin/env python
"""Analyse and plot the catastrophic-forgetting experiment.

Reads every ``forgetting_matrix.csv`` written by runs launched with
``--cf-experiment`` (see hnet_exp._eval_forgetting_matrix), aggregates over
seeds, and produces the figures + CL metrics that back the three claims:

  SAFE     — filtered KOZ violations stay ~0 for all (after_task, eval_task),
             including old tasks re-tested after new learning.
  FORGETS  — unfiltered KOZ violations on an OLD task rise as later tasks are
             learned (the shared PolicyNet overwrites its avoidance behaviour).
  USEFUL   — filter_saves = unfiltered_koz - filtered_koz grows on old tasks,
             i.e. the non-learned filter actively prevents the catastrophe.

Usage
-----
    python scripts/plot_forgetting.py --runs "runs/cf/**/forgetting_matrix.csv" \
        --out runs/cf/analysis

    # or point at the parent dir and let it glob:
    python scripts/plot_forgetting.py --runs runs/cf --out runs/cf/analysis

Outputs (in --out): matrix_koz.png, matrix_reward.png, retention_koz.png,
filter_saves.png, cl_metrics.png, and cl_metrics.csv (the numeric summary).
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colour-blind-safe, consistent per condition across every figure.
COND_COLOR = {"dagger": "#0072B2", "bc": "#D55E00"}
COND_LABEL = {"dagger": "DAgger", "bc": "BC-only"}


def load(runs_arg: str) -> pd.DataFrame:
    """Collect all forgetting_matrix.csv under the given path/glob into one frame."""
    if os.path.isdir(runs_arg):
        patterns = [os.path.join(runs_arg, "**", "forgetting_matrix.csv")]
    elif runs_arg.endswith(".csv"):
        patterns = [runs_arg]
    else:
        patterns = [runs_arg]

    files: list[str] = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"No forgetting_matrix.csv found under: {runs_arg}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["run"] = f
        frames.append(df)
        print(f"  loaded {len(df):3d} rows  {f}")
    out = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(out)} rows  |  conditions={sorted(out.condition.unique())}  "
          f"seeds={sorted(out.seed.unique())}  tasks={sorted(out.eval_task.unique())}")
    return out


def _matrix(df: pd.DataFrame, condition: str, filt: str, value: str) -> np.ndarray:
    """Mean-over-seeds matrix M[after_task, eval_task] of `value`. NaN above diag."""
    sub = df[(df.condition == condition) & (df["filter"] == filt)]
    if sub.empty:
        return None
    K = int(df.after_task.max()) + 1
    M = np.full((K, K), np.nan)
    g = sub.groupby(["after_task", "eval_task"])[value].mean()
    for (k, j), v in g.items():
        M[int(k), int(j)] = v
    return M


def plot_matrices(df: pd.DataFrame, out: str) -> None:
    """Heatmaps of the forgetting matrix (KOZ filtered/unfiltered, reward filtered)."""
    conds = [c for c in ("dagger", "bc") if c in df.condition.unique()]
    specs = [
        ("koz_mean", "unfiltered", "Unfiltered KOZ violations/ep", "Reds"),
        ("koz_mean", "filtered",   "Filtered KOZ violations/ep",   "Reds"),
    ]
    fig, axes = plt.subplots(len(conds), len(specs),
                             figsize=(5.2 * len(specs), 4.4 * len(conds)),
                             squeeze=False)
    for r, cond in enumerate(conds):
        for c, (val, filt, title, cmap) in enumerate(specs):
            ax = axes[r][c]
            M = _matrix(df, cond, filt, val)
            if M is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(M, cmap=cmap, origin="upper", aspect="equal",
                           vmin=0, vmax=np.nanmax(_matrix(df, cond, "unfiltered", val)))
            ax.set_title(f"{COND_LABEL[cond]} — {title}")
            ax.set_xlabel("evaluated on task j")
            ax.set_ylabel("after training task k")
            ax.set_xticks(range(M.shape[1]))
            ax.set_yticks(range(M.shape[0]))
            for k in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if not np.isnan(M[k, j]):
                        ax.text(j, k, f"{M[k, j]:.1f}", ha="center", va="center",
                                fontsize=8,
                                color="white" if M[k, j] > 0.5 * np.nanmax(M) else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Forgetting matrix — cell (k, j) = policy after task k, tested on task j",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out, "matrix_koz.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")

    # Reward matrix (filtered) — usefulness / performance retention.
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.4),
                             squeeze=False)
    for c, cond in enumerate(conds):
        ax = axes[0][c]
        M = _matrix(df, cond, "filtered", "reward")
        if M is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(M, cmap="viridis", origin="upper", aspect="equal")
        ax.set_title(f"{COND_LABEL[cond]} — filtered reward")
        ax.set_xlabel("evaluated on task j")
        ax.set_ylabel("after training task k")
        ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(M.shape[0]))
        for k in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[k, j]):
                    ax.text(j, k, f"{M[k, j]:.0f}", ha="center", va="center",
                            fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = os.path.join(out, "matrix_reward.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_retention(df: pd.DataFrame, out: str) -> None:
    """Per old-task retention curves: metric on task j as later tasks are learned."""
    tasks = sorted(df.eval_task.unique())
    conds = [c for c in ("dagger", "bc") if c in df.condition.unique()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: unfiltered KOZ on task 0 vs after_task (the headline forgetting curve).
    ax = axes[0]
    j0 = 0
    for cond in conds:
        for filt, ls, mark in (("unfiltered", "-", "o"), ("filtered", "--", "s")):
            sub = df[(df.condition == cond) & (df["filter"] == filt)
                     & (df.eval_task == j0)]
            if sub.empty:
                continue
            g = sub.groupby("after_task")["koz_mean"]
            ks = sorted(sub.after_task.unique())
            mean = g.mean().reindex(ks).values
            std = g.std().reindex(ks).fillna(0).values
            ax.plot(ks, mean, ls, marker=mark, color=COND_COLOR[cond],
                    label=f"{COND_LABEL[cond]} {filt}")
            ax.fill_between(ks, mean - std, mean + std, color=COND_COLOR[cond], alpha=0.12)
    ax.set_title(f"Retention of task {j0}: KOZ violations vs tasks learned since")
    ax.set_xlabel("after training task k")
    ax.set_ylabel(f"KOZ violations/ep on task {j0}")
    ax.set_xticks(sorted(df.after_task.unique()))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: average unfiltered KOZ over all PREVIOUS tasks (j < k) vs k.
    ax = axes[1]
    for cond in conds:
        for filt, ls, mark in (("unfiltered", "-", "o"), ("filtered", "--", "s")):
            sub = df[(df.condition == cond) & (df["filter"] == filt)]
            if sub.empty:
                continue
            prev = sub[sub.eval_task < sub.after_task]
            if prev.empty:
                continue
            # mean over seeds & prior tasks, per k
            per_seed = prev.groupby(["seed", "after_task"])["koz_mean"].mean().reset_index()
            g = per_seed.groupby("after_task")["koz_mean"]
            ks = sorted(per_seed.after_task.unique())
            mean = g.mean().reindex(ks).values
            std = g.std().reindex(ks).fillna(0).values
            ax.plot(ks, mean, ls, marker=mark, color=COND_COLOR[cond],
                    label=f"{COND_LABEL[cond]} {filt}")
            ax.fill_between(ks, mean - std, mean + std, color=COND_COLOR[cond], alpha=0.12)
    ax.set_title("Mean KOZ violations over ALL previously-learned tasks (j < k)")
    ax.set_xlabel("after training task k")
    ax.set_ylabel("mean KOZ violations/ep on old tasks")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out, "retention_koz.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def plot_filter_saves(df: pd.DataFrame, out: str) -> None:
    """filter_saves = unfiltered_koz - filtered_koz on old tasks, per k."""
    conds = [c for c in ("dagger", "bc") if c in df.condition.unique()]
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in conds:
        u = df[(df.condition == cond) & (df["filter"] == "unfiltered")]
        f = df[(df.condition == cond) & (df["filter"] == "filtered")]
        if u.empty or f.empty:
            continue
        merged = u.merge(
            f, on=["seed", "after_task", "eval_task", "condition"],
            suffixes=("_u", "_f"))
        merged = merged[merged.eval_task < merged.after_task]  # old tasks only
        if merged.empty:
            continue
        merged["saves"] = (merged["koz_mean_u"] - merged["koz_mean_f"]).clip(lower=0)
        per_seed = merged.groupby(["seed", "after_task"])["saves"].mean().reset_index()
        g = per_seed.groupby("after_task")["saves"]
        ks = sorted(per_seed.after_task.unique())
        mean = g.mean().reindex(ks).values
        std = g.std().reindex(ks).fillna(0).values
        ax.plot(ks, mean, "-o", color=COND_COLOR[cond], label=COND_LABEL[cond])
        ax.fill_between(ks, mean - std, mean + std, color=COND_COLOR[cond], alpha=0.15)
    ax.set_title("Catastrophe prevented by the filter on old tasks\n"
                 "(unfiltered − filtered KOZ violations/ep, averaged over j < k)")
    ax.set_xlabel("after training task k")
    ax.set_ylabel("filter saves (violations/ep)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out, "filter_saves.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")


def cl_metrics(df: pd.DataFrame, out: str) -> pd.DataFrame:
    """Standard continual-learning metrics per condition, aggregated over seeds.

    For metric m and matrix R[k][j] (policy after task k, tested on task j):
      ACC_final   = mean_j R[K][j]                 (final average performance)
      BWT         = mean_{j<K} R[K][j] - R[j][j]   (backward transfer)
      Forgetting  = mean_{j<K} R[K][j] - R[j][j]   (same sign convention here)
    Computed for filtered reward (higher=better) and unfiltered KOZ (lower=better),
    plus the safety-retention number = final filtered KOZ (should be ~0).
    """
    rows = []
    K = int(df.after_task.max())
    for cond in sorted(df.condition.unique()):
        for seed in sorted(df[df.condition == cond].seed.unique()):
            s = df[(df.condition == cond) & (df.seed == seed)]

            def R(filt, value):
                m = {}
                for _, r in s[s["filter"] == filt].iterrows():
                    m[(int(r.after_task), int(r.eval_task))] = r[value]
                return m

            for filt, value, name in (
                    ("filtered",   "reward",   "reward_filtered"),
                    ("unfiltered", "reward",   "reward_unfiltered"),
                    ("unfiltered", "koz_mean", "koz_unfiltered"),
                    ("filtered",   "koz_mean", "koz_filtered")):
                Rm = R(filt, value)
                final = [Rm[(K, j)] for j in range(K + 1) if (K, j) in Rm]
                bwt = [Rm[(K, j)] - Rm[(j, j)]
                       for j in range(K) if (K, j) in Rm and (j, j) in Rm]
                rows.append({
                    "condition": cond, "seed": seed, "metric": name,
                    "acc_final": np.mean(final) if final else np.nan,
                    "bwt": np.mean(bwt) if bwt else np.nan,
                })
    m = pd.DataFrame(rows)
    summary = (m.groupby(["condition", "metric"])[["acc_final", "bwt"]]
               .agg(["mean", "std"]).reset_index())
    csvp = os.path.join(out, "cl_metrics.csv")
    summary.to_csv(csvp, index=False)
    print(f"  wrote {csvp}")
    print("\n=== CL metrics (mean over seeds) ===")
    print(m.groupby(["condition", "metric"])[["acc_final", "bwt"]].mean().round(2))

    # Bar chart: final unfiltered vs filtered KOZ per condition (the headline).
    conds = sorted(df.condition.unique())
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conds))
    w = 0.35
    for i, (metric, off, lbl) in enumerate([
            ("koz_unfiltered", -w / 2, "unfiltered (raw policy)"),
            ("koz_filtered",   +w / 2, "filtered (with safety filter)")]):
        means, stds = [], []
        for cond in conds:
            sel = m[(m.condition == cond) & (m.metric == metric)]["acc_final"]
            means.append(sel.mean())
            stds.append(sel.std())
        ax.bar(x + off, means, w, yerr=stds, capsize=4, label=lbl,
               color="#D55E00" if "unfiltered" in metric else "#0072B2")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds])
    ax.set_ylabel("final average KOZ violations/ep (all tasks)")
    ax.set_title("Safety at end of the task sequence\n(lower is safer; filtered ≈ 0 = guarantee holds)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(out, "cl_metrics.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"  wrote {p}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True,
                    help="dir containing runs, or a glob of forgetting_matrix.csv files")
    ap.add_argument("--out", default="cf_analysis", help="output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading forgetting matrices from: {args.runs}")
    df = load(args.runs)

    plot_matrices(df, args.out)
    plot_retention(df, args.out)
    plot_filter_saves(df, args.out)
    cl_metrics(df, args.out)
    print(f"\nDone. Figures in {args.out}/")


if __name__ == "__main__":
    main()
