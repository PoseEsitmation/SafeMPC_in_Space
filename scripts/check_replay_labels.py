#!/usr/bin/env python
"""How good are expert-replay labels?  Offline, on a finished run.

    python scripts/check_replay_labels.py --run runs/cl_s5/cl_hnet_replay_s0 --device cpu

For every saved checkpoint model_k.pt and every task j <= k, a window of the
task's MPC-phase states is relabelled by the MPC expert planning through the
hypernetwork's weights for task j, three ways:

    cold   planner reset before every state (what refresh_replay does)
    cold2  the same again: the planner's own noise
    warm   planner reset once, then states in recorded order (how the stored
           labels were made, but with the model of checkpoint k)

and compared with the action the expert actually took when the task was
trained (stored).  Actions are in the physical box [-1, 1]^3.  k = j shows the
fresh teacher, k > j the teacher after later tasks.  cl_s5's replay arms did
worse than no replay: if cold >> warm here, cold-start labels are the cause.
"""
import argparse
import csv
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hypercrl import dataset  # noqa: E402  (data.pkl pickles hypercrl.dataset)
from hypercrl.control import SafeAgent  # noqa: E402
from hypercrl.envs.cl_env import CLEnvHandler  # noqa: E402
from hypercrl.envs.space_tasks import set_scenario_override  # noqa: E402
from hypercrl.model import build_model_hnet  # noqa: E402
from hypercrl.tools import HP, Hparams, reset_seed  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", required=True, help="run dir holding model/, data.pkl, hparams.csv")
ap.add_argument("--n", type=int, default=100, help="states per (checkpoint, task)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--ckpts", type=int, nargs="+", default=None, help="default: all model_k.pt")
args = ap.parse_args()

with open(os.path.join(args.run, "hparams.csv")) as f:
    saved = dict(csv.reader(f))
env_name, seed = saved["env"], int(saved["seed"])

hp = HP(env_name, seed, os.path.dirname(args.run), run_name=os.path.basename(args.run))
hp.model = "hnet"
hp.device = args.device
hp = Hparams.add_hnet_hparams(hp)
if saved.get("space_fixed_scenario") == "True":
    set_scenario_override(angle_bound_lower=120.0, angle_bound_upper=140.0,
                          half_angle_low_deg=20.0, half_angle_high_deg=20.0,
                          cone_offset_deg=10.0)
reset_seed(seed)

sys.modules["dataset"] = dataset
with open(os.path.join(args.run, "data.pkl"), "rb") as f:
    collector = pickle.load(f)

ckpts = args.ckpts if args.ckpts is not None else sorted(
    int(p[6:-3]) for p in os.listdir(os.path.join(args.run, "model"))
    if p.startswith("model_") and p.endswith(".pt"))
envs = CLEnvHandler(env_name, seed)
for t in range(max(ckpts) + 1):
    envs.add_task(t)


def gap(a, b):
    return (a - b).norm(dim=1).mean().item()


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=1).mean().item()


rows = []
print(f"{'ckpt':>4} {'task':>4} | {'|cold-stored|':>13} {'|warm-stored|':>13} "
      f"{'|cold-cold2|':>12} | {'cos cold':>8} {'cos warm':>8} | "
      f"{'|u| stored':>10} {'cold':>6} {'warm':>6}")
for k in ckpts:
    mnet, hnet = build_model_hnet(hp, num_input=2)
    ck = torch.load(os.path.join(args.run, "model", f"model_{k}.pt"),
                    map_location=args.device, weights_only=False)
    for t in range(ck["num_tasks_seen"]):
        hnet.add_task(t, hp.std_normal_temb)
    mnet.load_state_dict(ck["mnet_state_dict"])
    hnet.load_state_dict(ck["hnet_state_dict"])
    mnet.to(args.device), hnet.to(args.device)
    agent = SafeAgent(hp, mnet, hnet=hnet, collector=collector)

    for j in range(k + 1):
        env_j = envs.get_env(j)
        if hasattr(env_j, "get_safety_filter"):
            agent.set_safety_filter(env_j.get_safety_filter())
        agent.cache_hnet(j)

        X = np.hstack(collector.states[j]).T
        U = np.hstack(collector.actions[j]).T
        rng = np.random.default_rng(1000 * k + j)
        start = int(rng.integers(hp.init_rand_steps, X.shape[0] - args.n))
        X, U = X[start:start + args.n], torch.as_tensor(U[start:start + args.n]).float()

        def label(x, reset):
            if reset:
                agent.reset()
            return agent.act(x, task_id=j).detach().cpu().flatten().float()

        cold = torch.stack([label(x, True) for x in X])
        cold2 = torch.stack([label(x, True) for x in X])
        agent.reset()
        warm = torch.stack([label(x, False) for x in X])

        r = dict(ckpt=k, task=j, gap_cold=gap(cold, U), gap_warm=gap(warm, U),
                 gap_self=gap(cold, cold2), cos_cold=cos(cold, U), cos_warm=cos(warm, U),
                 norm_stored=U.norm(dim=1).mean().item(),
                 norm_cold=cold.norm(dim=1).mean().item(),
                 norm_warm=warm.norm(dim=1).mean().item())
        rows.append(r)
        print(f"{k:>4} {j:>4} | {r['gap_cold']:>13.3f} {r['gap_warm']:>13.3f} "
              f"{r['gap_self']:>12.3f} | {r['cos_cold']:>8.2f} {r['cos_warm']:>8.2f} | "
              f"{r['norm_stored']:>10.2f} {r['norm_cold']:>6.2f} {r['norm_warm']:>6.2f}",
              flush=True)

out = os.path.join(args.run, "replay_label_check.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}")
