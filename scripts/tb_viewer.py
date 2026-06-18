#!/usr/bin/env python3
"""
TensorBoard run selector and visualizer for SafeMPC space environment.

Usage:
    python scripts/tb_viewer.py                    # interactive CLI picker
    python scripts/tb_viewer.py --run <path>       # direct path
    python scripts/tb_viewer.py --save plot.png    # save instead of show

Overview:
    Reads TensorBoard event files from the runs/ directory and produces a
    scatter-plot dashboard for space-environment training runs. Episodes are
    colour-coded by outcome: normal (blue), KOZ-violated (magenta),
    non-settled (orange).

    Panels (spaceEnv only):
        1. Reward          — total episode reward
        2. KOZ violations  — keep-out zone crossings per episode
        3. Theta margin    — boresight distance from the KOZ boundary [deg]
        4. Control effort  — E(t_end) = ∫‖τ‖² dt [N²m²s]; shown only when
                             the tag train_env/task_N/control_effort is present
                             (logged from training runs after this metric was added)

    For non-space environments the script reports that space-env tags are
    missing and exits without plotting.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs")

# ── colours matching the image ───────────────────────────────────────────────
CAT_COLORS = {0: "#1f77b4", 1: "#e040fb", 2: "#ff9800"}   # blue / magenta / orange
CAT_LABELS = {0: "normal", 1: "violated", 2: "non-settled"}


# ── run discovery ─────────────────────────────────────────────────────────────

def find_runs(root):
    """Return list of directories that contain a tfevents file."""
    found = []
    for dirpath, _, files in os.walk(root):
        if any(f.startswith("events.out.tfevents") for f in files):
            found.append(dirpath)
    return sorted(found)


def pick_run_cli(root):
    runs = find_runs(root)
    if not runs:
        sys.exit(f"No TensorBoard runs found under {root}")

    print("\nAvailable TensorBoard runs:") #showing local tensorboard runs
    for i, r in enumerate(runs):
        rel = os.path.relpath(r, root)
        print(f"  [{i:2d}]  {rel}")

    while True:
        try:
            raw = input("\nEnter run number (or 'exit'): ").strip() #enter number [xx] next to tensorboard name
            if raw.lower() == "exit":
                sys.exit(0)
            idx = int(raw)
            if 0 <= idx < len(runs):
                return runs[idx]
        except (ValueError, KeyboardInterrupt):
            pass
        print("  Invalid selection, try again.")



# ── data loading ──────────────────────────────────────────────────────────────

def load_scalars(run_dir):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = {
            "steps":  np.array([e.step  for e in events]),
            "values": np.array([e.value for e in events]),
        }
    return data


# ── episode classification ────────────────────────────────────────────────────

def classify(koz_vals, time_vals, max_time_thresh=None):
    """
    Returns int array:  0 = normal,  1 = violated (KOZ),  2 = non-settled.
    non-settled = episode_time reached/exceeded the time limit.
    """
    if max_time_thresh is None:
        max_time_thresh = np.max(time_vals) if len(time_vals) else 200.0
    cats = np.zeros(len(koz_vals), dtype=int)
    cats[koz_vals > 0] = 1
    cats[time_vals >= max_time_thresh * 0.99] = 2
    return cats


# ── scatter helper ────────────────────────────────────────────────────────────

def scatter_panel(ax, x, y, cats, ylabel, ylim=None): #getting the "dot clouds"
    for cat in [0, 1, 2]:
        mask = cats == cat
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], s=6, alpha=0.65,
                   color=CAT_COLORS[cat], label=CAT_LABELS[cat], rasterized=True)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)


# ── space-env specific plot ───────────────────────────────────────────────────

def plot_space_eval(data, run_name, task="0", save_path=None):
    """Three-panel scatter using train_env (preferred, more points) or eval_env."""

    # ── prefer train_env (logged every 1000 steps) over eval_env (every 5000) ──
    train_pfx = f"train_env/task_{task}/"

    # theta_margin tag has a typo in older runs ("thetha")
    theta_tag = None
    for candidate in (train_pfx + "theta_margin", train_pfx + "thetha_margin"):
        if candidate in data:
            theta_tag = candidate
            break

    if not all(k in data for k in (train_pfx + "reward", train_pfx + "koz_violations")) \
            or not theta_tag:
        return False   # fall through to generic plot

    pfx    = train_pfx
    steps  = data[pfx + "reward"]["steps"]
    rews   = data[pfx + "reward"]["values"]
    koz    = data[pfx + "koz_violations"]["values"]

    # theta_margin may be logged at every step — sample only at reward's steps
    theta_steps  = data[theta_tag]["steps"]
    theta_values = data[theta_tag]["values"]
    step_to_idx  = {s: i for i, s in enumerate(theta_steps)}
    third_vals   = np.array([theta_values[step_to_idx[s]]
                             for s in steps if s in step_to_idx])
    steps = np.array([s for s in steps if s in step_to_idx])
    rews  = rews[:len(steps)]
    koz   = koz[:len(steps)]

    third_label = "Theta margin\n(rad)"
    third_ylim  = None
    source      = f"train_env  ({len(steps)} pts)"

    cats = np.zeros(len(koz), dtype=int)
    cats[koz > 0] = 1

    n       = len(steps)
    n_vio   = int((cats == 1).sum())
    vio_pct = n_vio / max(n, 1) * 100

    x = np.arange(len(steps))

    # control effort (optional — absent in runs logged before this metric was added)
    effort_tag = train_pfx + "control_effort"
    has_effort = effort_tag in data
    if has_effort:
        effort_steps  = data[effort_tag]["steps"]
        effort_values = data[effort_tag]["values"]
        effort_idx    = {s: i for i, s in enumerate(effort_steps)}
        effort_vals   = np.array([effort_values[effort_idx[s]]
                                  for s in steps if s in effort_idx])
        effort_x      = np.arange(len(effort_vals))
        effort_cats   = cats[:len(effort_vals)]

    nrows = 4 if has_effort else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 4 * nrows - 2), sharex=True)
    fig.suptitle(
        f"{os.path.basename(run_name)}  [{source}]\n"
        f"Violated rate: {vio_pct:.3f}%",
        fontsize=11,
    )

    scatter_panel(axes[0], x, rews,       cats, "Reward")
    scatter_panel(axes[1], x, koz,        cats, "KOZ violations\n(per episode)")
    scatter_panel(axes[2], x, third_vals, cats, third_label, ylim=third_ylim)

    if has_effort:
        scatter_panel(axes[3], effort_x, effort_vals, effort_cats,
                      "Control effort\n(N²m²s)")
        axes[3].set_xlabel("Sample Number")
    else:
        axes[2].set_xlabel("Training steps")

    axes[1].legend(loc="upper right", markerscale=2.5, fontsize=9)

    plt.tight_layout()
    _finish(fig, save_path)
    return True



def _finish(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize a TensorBoard run.")
    parser.add_argument("--run",  default=None, help="Path to run directory")
    parser.add_argument("--save", default=None, help="Save figure to file instead of showing")
    parser.add_argument("--task",   default="0",   help="Task index (default: 0)")
    args = parser.parse_args()

    # ── select run (loop until data is found or user exits) ──
    if args.run:
        run_dir = args.run
        print(f"\nLoading  {run_dir} …")
        data = load_scalars(run_dir)
        print(f"Found {len(data)} scalar tag(s).")
        if not data:
            sys.exit("No scalar data found.")
    else:
        while True:
            run_dir = pick_run_cli(RUNS_ROOT)
            if not run_dir:
                sys.exit("No run selected.")
            print(f"\nLoading  {run_dir} …")
            data = load_scalars(run_dir)
            print(f"Found {len(data)} scalar tag(s).")
            if data:
                break
            print("  No scalar data in this run — please select another (or type 'exit' to quit).")

    done = plot_space_eval(data, run_dir, task=args.task, save_path=args.save)
    if not done:
        print("Space-env eval tags not found in this run.")


if __name__ == "__main__":
    main()
