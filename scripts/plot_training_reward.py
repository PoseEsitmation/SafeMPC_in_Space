#!/usr/bin/env python
"""Per-episode training reward, plus steps held inside the 0.25 deg goal (the +9/step bonus)."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = [("cl_spaceEnv_null", "spaceEnv_null (control)", "#2a78d6"),
        ("cl_spaceEnv_thruster", "spaceEnv_thruster (faults)", "#eb6834")]
ROOT, NUM_TASKS, EPS_PER_TASK = "runs/cl_s1", 3, 12


def load(run):
    ea = EventAccumulator(glob.glob(f"{ROOT}/{run}/events*")[0], size_guidance={"scalars": 0})
    ea.Reload()
    rewards, goal_steps = [], []
    for t in range(NUM_TASKS):
        s = ea.Scalars(f"train_env/task_{t}/attitude_error_deg")
        steps, err = np.array([x.step for x in s]), np.array([x.value for x in s])
        prev = steps.min() - 1
        for e in ea.Scalars(f"train_env/task_{t}/reward"):
            in_ep = (steps > prev) & (steps <= e.step)
            goal_steps.append(int((err[in_ep] <= 0.25).sum()))
            rewards.append(e.value)
            prev = e.step
    return np.array(rewards), np.array(goal_steps)


fig, (ax_r, ax_g) = plt.subplots(2, 1, figsize=(12.5, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
for i, (run, label, color) in enumerate(RUNS):
    r, g = load(run)
    x = np.arange(len(r)) + (i - 0.5) * 0.4
    ax_r.bar(x, r, 0.37, color=color, label=label)
    ax_g.bar(x, g, 0.37, color=color)

for ax in (ax_r, ax_g):
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for b in range(1, NUM_TASKS):
        ax.axvline(b * EPS_PER_TASK - 0.5, color="grey", ls="--", lw=1)
ax_r.axhline(0, color="grey", lw=0.8)
ax_r.set_ylabel("episode reward")
ax_g.set_ylabel("steps inside\n0.25° goal")
ax_g.set_xlabel(f"training episode ({EPS_PER_TASK} per task)")
ax_r.set_title("Training reward per episode (tasks separated by dashed lines)", loc="left")
ax_r.legend(frameon=False, loc="upper left")

fig.tight_layout()
out = f"{ROOT}/analysis/training_reward.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("wrote", out)
