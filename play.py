#!/usr/bin/env python
"""Replay a saved HyperCRL run directory with full rendering.

Usage
-----
# Play all trained tasks:
python play.py <run_dir>

# Play a specific trained task:
python play.py <run_dir> --task 0

# Play an untrained task (uses last trained task's weights):
python play.py <run_dir> --task 3

# Play with custom episode count:
python play.py <run_dir> --task 0 --episodes 5

# Run directory looks for example like this:
runs/lqr/20260617_111148_TBcartpole_single_2020

The script reads hparams.csv + tasks.json from the run directory, loads the
saved model weights, runs the env in render mode, and writes a replay stats
CSV to {folder}/replay/{timestamp}_stats.csv.
"""

import argparse
import ast
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hypercrl.tools.default_arg import HP, Hparams
from hypercrl.model.tools import build_model, build_model_hnet
from hypercrl.control import SafeAgent
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.tools import reset_seed

HNET_MODELS = {"hnet", "chunked_hnet", "hnet_mt", "hnet_replay", "hnet_ewc", "hnet_si"}


# ---------------------------------------------------------------------------
# hparams reconstruction from saved CSV
# ---------------------------------------------------------------------------

def _parse_csv_value(val_str, current):
    if val_str == "None":
        return None
    if isinstance(current, bool) or val_str in ("True", "False"):
        return val_str == "True"
    if isinstance(current, int):
        return int(val_str)
    if isinstance(current, float):
        return float(val_str)
    if isinstance(current, list):
        return ast.literal_eval(val_str)
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    return val_str


def load_hparams(folder: str) -> Hparams:
    csv_path = os.path.join(folder, "hparams.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"hparams.csv not found in {folder!r}")

    raw: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            raw[row["config"]] = row["value"]

    env = raw["env"]
    seed = int(raw["seed"])
    save_folder = raw.get("save_folder", "./runs/lqr")

    hparams = HP(env, seed, save_folder)

    for key, val_str in raw.items():
        current = getattr(hparams, key, None)
        try:
            setattr(hparams, key, _parse_csv_value(val_str, current))
        except Exception:
            setattr(hparams, key, val_str)

    return hparams


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------

def _model_dir(folder: str) -> str:
    return os.path.join(folder, "model")


def load_checkpoint(folder: str, hparams: Hparams, task_id=None) -> dict:
    filename = "model.pt" if task_id is None else f"model_{task_id}.pt"
    path = os.path.join(_model_dir(folder), filename)
    if not os.path.isfile(path):
        available = sorted(os.listdir(_model_dir(folder)))
        raise FileNotFoundError(
            f"No checkpoint at {path!r}.\nAvailable: {available}")
    print(f"[play] loading  {path}")
    return torch.load(path, map_location=hparams.device, weights_only=False)


# ---------------------------------------------------------------------------
# agent construction
# ---------------------------------------------------------------------------

def build_agent(hparams: Hparams, checkpoint: dict) -> SafeAgent:
    if hparams.model in HNET_MODELS:
        mnet, hnet = build_model_hnet(hparams)
        for tid in range(checkpoint["num_tasks_seen"]):
            hnet.add_task(tid, hparams.std_normal_temb)
        mnet.load_state_dict(checkpoint["mnet_state_dict"])
        hnet.load_state_dict(checkpoint["hnet_state_dict"])
        mnet.to(hparams.device)
        hnet.to(hparams.device)
        print(f"[play] hnet restored  ({checkpoint['num_tasks_seen']} task(s))")
        return SafeAgent(hparams, mnet, hnet=hnet)
    else:
        model = build_model(hparams)
        for tid in range(checkpoint["num_tasks_seen"]):
            model.add_weights(tid)
        if hparams.model == "ewc":
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        model.to(hparams.device)
        print(f"[play] model restored  ({checkpoint['num_tasks_seen']} task(s))")
        return SafeAgent(hparams, model)


def restore_norms(agent: SafeAgent, checkpoint: dict, task_id: int,
                  device: str, folder: str) -> None:
    """Push saved normalization stats onto the agent for the given task.

    The dynamics model was trained on normalised (x, u) inputs.  Without these
    stats the agent receives raw values it has never seen → garbage actions.

    Tries in order:
      1. norms embedded in the checkpoint (new format)
      2. data.pkl in the run folder (end-of-task save, older format)
    """
    norms = checkpoint.get("norms", {})

    if task_id not in norms:
        # Fallback: try loading from data.pkl (written by end-of-task save)
        pkl_path = os.path.join(folder, "data.pkl")
        if os.path.isfile(pkl_path):
            import pickle, sys
            from hypercrl import dataset as _ds_module
            sys.modules.setdefault("dataset", _ds_module)
            with open(pkl_path, "rb") as f:
                collector = pickle.load(f)
            try:
                x_mu, x_std, a_mu, a_std = collector.norm(task_id)
                norms = {task_id: {"x_mu": x_mu.cpu(), "x_std": x_std.cpu(),
                                   "a_mu": a_mu.cpu(), "a_std": a_std.cpu()}}
                print(f"[play] norm stats for task {task_id} loaded from data.pkl")
            except (KeyError, IndexError):
                pass

    if task_id not in norms:
        print(f"[play] WARNING: no norm stats for task {task_id} — "
              "model was trained with normalisation enabled but stats are missing. "
              "Re-run training with the current code to fix this.")
        return

    n = norms[task_id]
    mpc = agent.mpc
    mpc.normalize_xu = True
    mpc.x_mu = n["x_mu"].to(device)
    mpc.x_std = n["x_std"].to(device)
    mpc.a_mu = n["a_mu"].to(device)
    mpc.a_std = n["a_std"].to(device)


# ---------------------------------------------------------------------------
# stats saving
# ---------------------------------------------------------------------------

def save_replay_stats(folder: str, rows: list[dict]) -> str:
    replay_dir = os.path.join(folder, "replay")
    os.makedirs(replay_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(replay_dir, f"{ts}_stats.csv")
    if not rows:
        return out_path
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[play] stats → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# play loop
# ---------------------------------------------------------------------------

def play(folder: str, task: int = None, episodes: int = 10) -> None:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Run directory not found: {folder!r}")

    hparams = load_hparams(folder)
    hparams.device = (
        "cuda:0" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Pick checkpoint file: per-task snapshot when available, else model.pt
    ckpt_task_id = None
    if task is not None:
        task_pt = os.path.join(_model_dir(folder), f"model_{task}.pt")
        ckpt_task_id = task if os.path.isfile(task_pt) else None

    checkpoint = load_checkpoint(folder, hparams, task_id=ckpt_task_id)

    num_seen = checkpoint["num_tasks_seen"]
    if num_seen == 0:
        print("[play] mid-run checkpoint — no completed task yet, playing as task 0")
        num_seen = 1

    tasks_to_play = [task] if task is not None else list(range(num_seen))

    # Print task manifest if available
    tasks_json = os.path.join(folder, "tasks.json")
    if os.path.isfile(tasks_json):
        with open(tasks_json) as f:
            manifest = json.load(f)
        print("[play] task manifest:")
        for t in manifest[:num_seen]:
            print(f"       {t}")
    else:
        print("[play] (no tasks.json — env reconstructed from hparams)")

    print(f"\n[play] env={hparams.env}  model={hparams.model}  seed={hparams.seed}")
    print(f"[play] tasks: {tasks_to_play}   episodes each: {episodes}\n")

    reset_seed(hparams.seed)
    agent = build_agent(hparams, checkpoint)

    envs = CLEnvHandler(hparams.env, hparams.seed)
    stat_rows: list[dict] = []

    for tid in tasks_to_play:
        # For single-task model, reload per-task weights when they exist
        if hparams.model == "single" and num_seen > 1:
            task_pt = os.path.join(_model_dir(folder), f"model_{tid}.pt")
            if os.path.isfile(task_pt):
                checkpoint = load_checkpoint(folder, hparams, task_id=tid)
                agent = build_agent(hparams, checkpoint)

        weights_task = tid if tid < num_seen else num_seen - 1 #using weights from last training task

        if weights_task != tid:
            print(f"[play] task {tid} is untrained — using task {weights_task} weights")

        # Restore normalization stats for this task (critical: model was
        # trained on normalised inputs; without this, MPC produces bad actions)
        restore_norms(agent, checkpoint, tid, hparams.device, folder)

        for skip_id in range(len(envs._envs), tid): # skipping "missing task" (trained on task 0 and want to play on task 3)
            envs.add_task(skip_id, render=False)
        env = envs.add_task(tid, render=True)
        print(f"--- Task {tid} ---")

        for ep in range(episodes):
            rewards = []
            x_t, _ = env.reset()
            agent.reset()
            done = False

            while not done:
                env.render()
                u_t = agent.act(x_t, task_id=weights_task).detach().cpu().numpy()
                x_tt, reward, terminated, truncated, _ = env.step(
                    u_t.reshape(env.action_space.shape))
                done = terminated or truncated
                x_t = x_tt
                rewards.append(reward)

            ep_reward = float(np.sum(rewards))
            ep_len = len(rewards)
            print(f"  ep {ep + 1}/{episodes}  reward={ep_reward:.1f}  steps={ep_len}")
            stat_rows.append({
                "task": tid,
                "episode": ep,
                "reward": ep_reward,
                "steps": ep_len,
            })

        task_rewards = [r["reward"] for r in stat_rows if r["task"] == tid]
        print(f"  → mean={np.mean(task_rewards):.1f}  std={np.std(task_rewards):.1f}\n")

    envs.close()
    save_replay_stats(folder, stat_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Replay a saved HyperCRL run (always rendered)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "folder",
        help="Path to the run directory (contains hparams.csv and model/)")
    ap.add_argument(
        "--task", type=int, default=None,
        help="Task index to replay (default: all completed tasks)")
    ap.add_argument(
        "--episodes", type=int, default=10,
        help="Episodes per task (default: 10)")
    args = ap.parse_args()

    play(args.folder, task=args.task, episodes=args.episodes)
