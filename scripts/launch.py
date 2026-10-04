#!/usr/bin/env python
"""Launch continual-learning runs: one process per (arm, seed), N at a time.

    conda activate safempc
    python scripts/launch.py                          # 3 arms x seed 0, full profile
    python scripts/launch.py --seeds 0 1 2            # 9 runs, 4 at a time
    python scripts/launch.py --devices cuda:0 cuda:1  # spread across GPUs
    python scripts/launch.py --profile fast           # shortened runs (~1/3 the time)
    python scripts/launch.py --env spaceEnv_conflict --first-task-mult 2   # conflicting faults

Keep it alive on a server:
    nohup python scripts/launch.py --seeds 0 1 2 > launch.log 2>&1 &
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

ARMS = {
    "hnet":         [],                             # hypernetwork, no replay
    "noreg":        ["--no-hnet-reg"],              # regulariser off, no replay
    "hnet_replay":  ["--replay"],                   # the method
    "noreg_replay": ["--replay", "--no-hnet-reg"],  # replay without the memory
    "hnet_replay_stored": ["--replay", "--replay-labels", "stored"],  # rehearsal, no relabelling
}

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--arms", nargs="+", default=["hnet_replay", "noreg_replay", "hnet"],
                choices=list(ARMS))
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--tasks", type=int, default=3)
ap.add_argument("--env", default="spaceEnv_thruster")
ap.add_argument("--out", default="runs/cl_s4")
ap.add_argument("--devices", nargs="+", default=["cuda:0"])
ap.add_argument("--max-parallel", type=int, default=4)
ap.add_argument("--profile", choices=["full", "fast"], default="full")
ap.add_argument("--replay-frac", type=float, default=0.2,
                help="sampling share of all replayed old tasks (replay arms); "
                     "cl_s5 used the balanced default 1/(k+1), i.e. 1/2-2/3")
ap.add_argument("--reg-share", type=float, default=0.8,
                help="hnet regulariser's target share of the total loss (0 = fixed beta)")
ap.add_argument("--first-task-mult", type=float, default=1.0,
                help="run task 0 this many times longer")
ap.add_argument("--wait-for-idle", action="store_true",
                help="wait until no other training is running before starting")
args = ap.parse_args()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Every run uses this interpreter.  Launched from conda's base, all of cl_s4's
# noreg runs and cl_s5's seeds 1-2 died at import ("No module named numpy").
pre = subprocess.run([sys.executable, "-c", "import numpy, torch, hypercrl"],
                     cwd=ROOT, capture_output=True, text=True)
if pre.returncode != 0:
    sys.exit(f"{sys.executable} can't import the project:\n"
             f"{pre.stderr.strip().splitlines()[-1]}\n"
             f"activate the right env first (conda activate safempc)")

os.makedirs(os.path.join(ROOT, args.out), exist_ok=True)

jobs = [(arm, seed) for seed in args.seeds for arm in args.arms]
devices = itertools.cycle(args.devices)

print(f"{len(jobs)} run(s), {args.max_parallel} at a time, profile={args.profile}")
print(f"~{10 if args.profile == 'full' else 4} h per run alone; "
      f"expect longer when several share a GPU\n")

if args.wait_for_idle:
    while subprocess.run(["pgrep", "-f", "main.py run"], capture_output=True).returncode == 0:
        print("waiting for running training to finish ...", flush=True)
        time.sleep(300)

running = []   # (Popen, name, logfile)
queue = list(jobs)
failed = []

while queue or running:
    while queue and len(running) < args.max_parallel:
        arm, seed = queue.pop(0)
        name = f"cl_{arm}_s{seed}"
        log = os.path.join(ROOT, args.out, f"{arm}_seed{seed}.log")
        cmd = [sys.executable, "-u", "main.py", "run", "--method", "hnet",
               "--env", args.env, "--device", next(devices), "--seed", str(seed),
               "--num-tasks", str(args.tasks), "--savepath", args.out,
               "--cf-experiment", "--fixed-scenario", "--name", name,
               "--reg-share", str(args.reg_share),
               "--first-task-mult", str(args.first_task_mult)] + ARMS[arm]
        if "--replay" in ARMS[arm]:
            cmd += ["--replay-frac", str(args.replay_frac)]
        if args.profile == "fast":
            cmd.append("--cl-profile")
        fh = open(log, "w")
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        running.append((p, name, fh))
        print(f"[{time.strftime('%H:%M')}] start {name} (pid {p.pid}) -> {log}", flush=True)
        time.sleep(10)   # stagger CUDA start-up

    time.sleep(30)
    for entry in list(running):
        p, name, fh = entry
        if p.poll() is not None:
            fh.close()
            running.remove(entry)
            status = "ok" if p.returncode == 0 else f"FAILED ({p.returncode})"
            if p.returncode != 0:
                failed.append(name)
            print(f"[{time.strftime('%H:%M')}] done  {name}: {status} "
                  f"({len(queue)} queued, {len(running)} running)", flush=True)

print(f"\nall finished. failed: {failed or 'none'}")
print(f"analyse:  python scripts/plot_forgetting.py --runs {args.out} --out {args.out}/analysis")
