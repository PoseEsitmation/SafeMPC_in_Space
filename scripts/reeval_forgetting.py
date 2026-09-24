#!/usr/bin/env python
"""Re-run the forgetting-matrix evaluation on a finished run's saved policies.

    python scripts/reeval_forgetting.py --run runs/cl_s1/cl_spaceEnv_thruster \
        --env spaceEnv_thruster --seed 0 --num-tasks 3 --out runs/cl_s2/reeval_thruster_s0
"""
import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from hypercrl.control.agent import NNPolicyAgent
from hypercrl.control.policy_net import PolicyNet
from hypercrl.dataset.datautil import DataCollector
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.hnet_exp import _eval_forgetting_matrix
from hypercrl.tools import HP

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--env", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--num-tasks", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--eps-filtered", type=int, default=None)
ap.add_argument("--eps-unfiltered", type=int, default=None)
args = ap.parse_args()

hp = HP(args.env, seed=args.seed)
hp.device = args.device
hp.model = "hnet"
hp.forget_eval_eps_filtered = args.eps_filtered or hp.forget_eval_eps_filtered
hp.forget_eval_eps_unfiltered = args.eps_unfiltered or hp.forget_eval_eps_unfiltered

collector = DataCollector(hp)
norms = torch.load(os.path.join(args.run, "norms.pt"), weights_only=True)
collector.load_frozen_norms(norms["norms"], norms.get("diff_norms"))
envs = CLEnvHandler(args.env, args.seed, seed_offset=500_000)
for t in range(args.num_tasks):
    collector.freeze_norms(t)
    envs.add_task(t)

policy = PolicyNet(hp.state_dim, hp.control_dim).to(hp.device)
agent = NNPolicyAgent(hp, policy, collector=collector)
os.makedirs(args.out, exist_ok=True)
writer = SummaryWriter(args.out)
for k in range(args.num_tasks):
    policy.load_state_dict(torch.load(os.path.join(args.run, "model", f"policy_{k}.pt"),
                                      map_location=hp.device, weights_only=True))
    _eval_forgetting_matrix(agent, envs, args.out, writer, k, hp)
writer.close()
