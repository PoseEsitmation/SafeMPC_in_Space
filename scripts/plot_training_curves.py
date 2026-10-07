#!/usr/bin/env python
"""Plot losses and rewards over training time for a continual-learning run dir.

    python scripts/plot_training_curves.py --runs runs/cl_c6 --out runs/cl_c6/analysis

Reads the TensorBoard events of every <dir>/cl_<arm>_s<seed>/ and averages
seeds per arm.  Each scalar family is logged on its own counter (dynamics
batches, MPC env steps, policy steps), so all are mapped onto one time axis,
MPC env steps, piecewise-linearly per task: the counters' task boundaries are
read from the run (first step a task's scalars appear), not assumed.

Writes:
    curves_hnet_loss.png      task loss, regulariser loss, regulariser share
    curves_model_memory.png   dynamics-model validation loss on each task
    curves_reward_teacher.png MPC expert reward on each task
    curves_reward_student.png student reward on each task: curve while it is
                              trained, markers at later task boundaries
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Fixed per arm on every chart (validated categorical slots 1-5, light mode).
ARMS = {
    "hnet":               ("hnet",              "#2a78d6"),
    "noreg":              ("noreg",             "#eb6834"),
    "hnet_replay":        ("hnet_replay",       "#1baf7a"),
    "noreg_replay":       ("noreg_replay",      "#eda100"),
    "hnet_replay_stored": ("hnet_replay_stored", "#e87ba4"),
}
INK, INK2, GRID, SURF = "#15233F", "#52514e", "#e4e2dc", "#fffffe"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 15, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 17, "axes.titlecolor": INK, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})


def load(run_dir):
    f = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))[-1]
    ea = EventAccumulator(f, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    return {t: pd.Series({e.step: e.value for e in ea.Scalars(t)}) for t in tags}


def first_step(s, tag):
    return s[tag].index.min() if tag in s else None


def boundaries(s, n_tasks, fmt):
    """Counter value at which each task starts, plus the last value."""
    b = [0]
    for j in range(1, n_tasks):
        b.append(first_step(s, fmt.format(j)))
    return b


def mapper(src, dst):
    """Piecewise-linear map from one counter's task boundaries to env steps."""
    return lambda x: np.interp(x, src, dst)


def run_frames(root):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "cl_*_s[0-9]*"))):
        m = re.match(r"cl_(.+)_s(\d+)$", os.path.basename(d))
        if not m or m.group(1) not in ARMS:
            continue
        s = load(d)
        n = 1 + max(int(t.split("/")[1].split("_")[1]) for t in s if t.startswith("eval_env/task_"))
        # env-step boundaries: where each task's training-env scalars start
        env_b = [0] + [first_step(s, f"train_env/task_{j}/attitude_error_deg") - 1
                       for j in range(1, n)]
        env_end = s[f"train_env/task_{n-1}/attitude_error_deg"].index.max()
        env_b.append(env_end)
        # dynamics batches: task j's validation loss appears soon after its start
        tr_end = s["train/loss"].index.max() + 1
        tr_b = [0] + [round(env_b[j] / env_end * tr_end) for j in range(1, n)] + [tr_end]
        # policy steps: first DAgger eval of each task
        po_b = [0] + [first_step(s, f"dagger_eval_filtered/task_{j}/reward")
                      for j in range(1, n)]
        po_b.append(s[f"dagger_eval_filtered/task_{n-1}/reward"].index.max())
        rows.append(dict(arm=m.group(1), seed=int(m.group(2)), s=s, n=n, env_b=env_b,
                         tr=mapper(tr_b, env_b), po=mapper(po_b, env_b), env=lambda x: x))
    return rows


def mean_curve(runs, tag, xmap, smooth=1):
    """Average the seeds of one arm on the env-step axis."""
    parts = []
    for r in runs:
        if tag not in r["s"]:
            continue
        v = r["s"][tag].sort_index()
        if smooth > 1:
            v = v.rolling(smooth, min_periods=1).mean()
        parts.append(pd.Series(v.values, index=np.round(r[xmap](v.index.values), -2)))
    if not parts:
        return None
    df = pd.concat(parts, axis=1)
    return df.mean(axis=1).sort_index()


def decorate(ax, env_b, n, labels=True):
    for j in range(1, n):
        ax.axvline(env_b[j], color=INK2, lw=1, ls=(0, (4, 4)))
    if labels:
        for j in range(n):
            ax.text((env_b[j] + env_b[j + 1]) / 2, 1.0, f"task {j}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", color=INK2, fontsize=13)
    ax.set_xlim(0, env_b[-1])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))


def by_arm(rows):
    out = {}
    for r in rows:
        out.setdefault(r["arm"], []).append(r)
    return [(a, out[a]) for a in ARMS if a in out]


def legend(fig, arms):
    handles = [plt.Line2D([], [], color=ARMS[a][1], lw=3) for a, _ in arms]
    fig.legend(handles, [ARMS[a][0] for a, _ in arms], loc="lower center",
               ncol=len(arms), frameon=False, fontsize=14, bbox_to_anchor=(0.5, -0.01))


def plot_losses(rows, out):
    arms = by_arm(rows)
    ref = rows[0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharex=True)
    specs = [("train/loss", "Task loss (dynamics model)", None),
             ("train/regularizer", "Regulariser loss (memory penalty)", None),
             ("train/reg_share", "Regulariser share of total loss", "pct")]
    for ax, (tag, title, kind) in zip(axes, specs):
        curves = []
        for a, rs in arms:
            c = mean_curve(rs, tag, "tr", smooth=5)
            if c is None:
                continue
            if kind == "pct":
                c = c * 100
            ax.plot(c.index, c.values, color=ARMS[a][1])
            curves.append((ARMS[a][0], ARMS[a][1], c))
        if kind == "pct":
            ax.axhline(80, color=INK, lw=1, ls=":")
            ax.text(ref["env_b"][-1] * 0.02, 82, "target 80%", color=INK, fontsize=13)
            ax.set_ylim(-3, 100)
            ax.set_ylabel("%")
        else:
            ax.set_yscale("log")
        ax.set_title(title, pad=26)
        decorate(ax, ref["env_b"], ref["n"])
        ax.set_xlabel("training time (MPC environment steps)")
    legend(fig, arms)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, "curves_hnet_loss.png"), dpi=110)
    plt.close(fig)


def per_task_panels(rows, out, fname, tag_fmt, xmap, title_fmt, ylabel, smooth=1, extra=None):
    arms = by_arm(rows)
    ref = rows[0]
    n = ref["n"]
    fig, axes = plt.subplots(1, n, figsize=(18, 6.2), sharex=True, sharey=True)
    names = ["nominal", "roll reversed", "roll ↔ yaw swapped"]
    for j, ax in enumerate(axes):
        curves = []
        for a, rs in arms:
            c = mean_curve(rs, tag_fmt.format(j), xmap, smooth=smooth)
            if c is None:
                continue
            ax.plot(c.index, c.values, color=ARMS[a][1])
            curves.append((ARMS[a][0], ARMS[a][1], c))
            if extra:
                extra(ax, a, rs, j)
        ax.axvspan(ref["env_b"][j], ref["env_b"][j + 1], color="#2a78d6", alpha=0.06, lw=0)
        ax.set_title(title_fmt.format(j=j, name=names[j] if j < len(names) else ""), pad=26)
        decorate(ax, ref["env_b"], n)
        ax.set_xlabel("training time (MPC environment steps)")
    axes[0].set_ylabel(ylabel)
    legend(fig, arms)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(out, fname), dpi=110)
    plt.close(fig)


def forgetting_points(root):
    files = glob.glob(os.path.join(root, "*", "forgetting_matrix.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])
    df["arm"] = df.condition.str.replace("^dagger_?", "", regex=True)
    df["arm"] = df.arm.replace({"": "hnet", "replay": "hnet_replay",
                                "replay_stored": "hnet_replay_stored"})
    return df[df["filter"] == "filtered"].groupby(["arm", "after_task", "eval_task"]).reward.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = run_frames(args.runs)
    print(f"{len(rows)} runs: " + ", ".join(f"{a}×{len(rs)}" for a, rs in by_arm(rows)))

    plot_losses(rows, args.out)
    per_task_panels(rows, args.out, "curves_model_memory.png", "val/task_{}/loss", "tr",
                    "Task {j} · {name}", "dynamics-model validation loss", smooth=3)
    per_task_panels(rows, args.out, "curves_reward_teacher.png", "eval_env/task_{}/reward", "env",
                    "Task {j} · {name}", "teacher (MPC) episode reward")

    pts = forgetting_points(args.runs)

    def marks(ax, arm, rs, j):
        # Re-tests after each later task, joined so the drop reads as a line.
        ref = rs[0]
        ks = [k for k in range(j, ref["n"]) if (arm, k, j) in pts.index]
        xs = [ref["env_b"][k + 1] for k in ks]
        ys = [pts[(arm, k, j)] for k in ks]
        ax.plot(xs, ys, ls=(0, (3, 3)), lw=1.5, color=ARMS[arm][1], zorder=4)
        ax.plot(xs, ys, "o", ms=9, color=ARMS[arm][1], mec=SURF, mew=2, zorder=5)

    per_task_panels(rows, args.out, "curves_reward_student.png",
                    "dagger_eval_filtered/task_{}/reward", "po",
                    "Task {j} · {name}", "student episode reward (filter on)", smooth=3,
                    extra=marks)
    print("wrote", ", ".join(sorted(p for p in os.listdir(args.out) if p.startswith("curves_"))))


if __name__ == "__main__":
    main()
