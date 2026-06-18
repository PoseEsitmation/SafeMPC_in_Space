#!/usr/bin/env python3
"""
TensorBoard run selector: scatter-plots every scalar tag.

Usage:
    python scripts/tensorboard_visual.py                    # interactive CLI picker
    python scripts/tensorboard_visual.py --run <path>       # direct path
    python scripts/tensorboard_visual.py --save plot.png    # save instead of show
"""

import argparse
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS_ROOT = os.path.join(os.path.dirname(__file__), "..", "runs")


# ── run discovery ─────────────────────────────────────────────────────────────

def find_runs(root):
    found = []
    for dirpath, _, files in os.walk(root):
        if any(f.startswith("events.out.tfevents") for f in files):
            found.append(dirpath)
    return sorted(found)


def pick_run_cli(root):
    runs = find_runs(root)
    if not runs:
        sys.exit(f"No TensorBoard runs found under {root}")

    print("\nAvailable TensorBoard runs:")
    for i, r in enumerate(runs):
        rel = os.path.relpath(r, root)
        print(f"  [{i:2d}]  {rel}")

    while True:
        try:
            raw = input("\nEnter run number (or 'exit'): ").strip()
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


# ── generic all-tags scatter ──────────────────────────────────────────────────

def plot_all_tags(data, run_name, save_path=None):
    tags = sorted(data.keys())
    n = len(tags)
    if n == 0:
        print("No scalar tags to plot.")
        return

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle(os.path.basename(run_name), fontsize=12)

    for i, tag in enumerate(tags):
        ax = axes[i // ncols][i % ncols]
        steps  = data[tag]["steps"]
        values = data[tag]["values"]
        ax.scatter(steps, values, s=4, alpha=0.6, rasterized=True)
        ax.set_title(tag, fontsize=8)
        ax.set_xlabel("Step", fontsize=7)
        ax.set_ylabel(tag.split("/")[-1], fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scatter-plot all tags in a TensorBoard run.")
    parser.add_argument("--run",  default=None, help="Path to run directory")
    parser.add_argument("--save", default=None, help="Save figure to file instead of showing")
    args = parser.parse_args()

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

    plot_all_tags(data, run_dir, save_path=args.save)


if __name__ == "__main__":
    main()
