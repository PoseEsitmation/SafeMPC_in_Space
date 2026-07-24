# SafeMPC in Space

Model-based continual learning for satellite attitude control with safety guarantees.

The agent learns a dynamics model using **HyperCRL** (a hypernetwork that generates
task-specific weights without forgetting previous tasks) and plans actions via **MPC**.
A **CBF/CLF safety filter** intercepts every proposed action and solves a small QP to
guarantee the satellite never enters the keep-out zone (KOZ), regardless of what the
learned model predicts.

## Repository layout

```
SafeMPC_in_Space/
├── main.py                       # CLI entry point
├── play.py                       # checkpoint replayer
├── hypercrl/
│   ├── envs/space_KOZ.py         # satellite attitude env with KOZ
│   ├── control/safety_filter.py  # CBF/CLF-QP filter
│   ├── control/agent.py          # MPC agent
│   ├── hypercl/                  # hypernetwork (hnet/mnet)
│   └── model/                    # dynamics model training
├── scripts/                      # analysis and batch-run helpers
└── DOCS/                         # component and API definitions
```

## Installation

**Prerequisites:** Python 3.12+, conda

```bash
conda create -n <your-env-name>
conda activate <your-env-name>
pip install -r requirements.txt
```
## Usage

### Basic Usage

```bash
python main.py run --method METHOD --env ENV
```

### Arguments

| Argument      | Required | Description                                               |
| ------------- | -------- | --------------------------------------------------------- |
| `--method`    | yes      | Learning method (see below)                               |
| `--env`       | yes      | Environment (see below)                                   |
| `--device`    | no       | `cpu`, `cuda`, `cuda:0`, `mps` (auto-detected if omitted) |
| `--seed`      | no       | Integer seed for reproducibility                          |
| `--savepath`  | no       | Path to save logs/checkpoints                             |
| `--rendering` | no       | Show the simulation window during training                |
| `--play`      | no       | Reload a saved checkpoint and replay the agent            |

### Methods

| Method         | Description                      |
| -------------- | -------------------------------- |
| `single`       | Single-task training baseline    |
| `finetune`     | Naive fine-tuning baseline       |
| `ewc`          | Elastic Weight Consolidation     |
| `si`           | Synaptic Intelligence            |
| `coreset`      | Coreset-based continual learning |
| `pnn`          | Progressive Neural Networks      |
| `multitask`    | Multitask learning baseline      |
| `hnet`         | Hypernetwork (HyperCRL)          |
| `chunked_hnet` | Chunked hypernetwork variant     |
| `hnet_mt`      | Multitask hypernetwork           |
| `hnet_replay`  | Hypernetwork with replay buffer  |

### Environments

| Environment         | Description                       |
| ------------------- | --------------------------------- |
| `cartpole`          | Cartpole balancing                |
| `half_cheetah_body` | Half-cheetah with body variations |
| `pusher`            | Pusher manipulation task          |
| `door_pose`         | Door opening task                 |
| `spaceEnv`          | Satellite attitude control with KOZ — 4 tasks varying difficulty and thruster strength |
| `spaceEnv_moi`      | Satellite attitude control with KOZ — 4 tasks varying moment of inertia tensor        |

**`spaceEnv` tasks:**

| Task | Name | What varies |
|------|------|-------------|
| 0 | default | 80–180° initial error, full torque (2 Nm), standard KOZ penalty |
| 1 | easy | 10–45° initial error |
| 2 | hard | 90–180° initial error + 5× stronger KOZ penalty |
| 3 | weak | Half thruster power (1 Nm) |

**`spaceEnv_moi` tasks:**

| Task | Description | Inertia `diag(Ixx, Iyy, Izz)` [kg·m²] |
|------|-------------|----------------------------------------|
| 0 | Asymmetric (baseline) | `(60, 50, 70)` |
| 1 | Nearly symmetric, small satellite | `(20, 22, 25)` |
| 2 | Heavy asymmetric, large satellite | `(120, 90, 150)` |
| 3 | Oblate flat-disk shape | `(80, 80, 20)` |

### Examples

```bash
# Basic run
python main.py run --method single --env cartpole --device cpu

# With rendering
python main.py run --method single --env cartpole --device cpu --rendering

# Specific GPU, fixed seed, save logs
python main.py run --method hnet --env half_cheetah_body --device cuda:0 --seed 42 --savepath ./logs

# Replay a trained checkpoint
python main.py run --method ewc --env cartpole --play

# Train HyperCRL on satellite attitude control with KOZ
python main.py run --method hnet --env spaceEnv --device cpu --seed 42 --savepath ./runs/space

# Replay trained checkpoint
python main.py run --method hnet --env spaceEnv --seed 42 --savepath ./runs/space --play

# --play finds the most recent run for the given env/method/seed combination.
# To replay a specific run, pass its full timestamped path as --savepath:
python main.py run --method hnet --env spaceEnv --savepath ./runs/space/20260617_143022_TBspaceEnv_hnet_42 --play

# Alternatively, use the standalone replayer to point directly at a checkpoint:
python play.py --savepath ./runs/space/20260617_143022_TBspaceEnv_hnet_42 --env spaceEnv
```

### TensorBoard

Training metrics are logged automatically. To view them:

**Step 1 — Start training** (Terminal 1):
```bash
python main.py run --method METHOD --env ENV --savepath ./runs/my_experiment
# e.g.: python main.py run --method single --env cartpole --device cpu --seed 42 --savepath ./runs/cartpole
```

Each run creates a timestamped subdirectory inside `--savepath`:
```
runs/my_experiment/
└── 20260617_143022_TBcartpole_single_2020/   ← one directory per run
    ├── events.out.tfevents.*                  ← TensorBoard data
    ├── hparams.csv                            ← hyperparameter snapshot
    ├── data.pkl                               ← replay buffer
    ├── cartpole_single_2020.csv              ← validation stats
    └── model/
        ├── model.pt                           ← latest checkpoint (overwritten every 1000 steps)
        ├── model_0.pt                         ← permanent end-of-task snapshots
        └── model_1.pt
```


| File | Behaviour |
|------|-----------|
| `model.pt` | Overwritten every `save_every` steps (default 1000) — crash-safe |
| `model_N.pt` | Written once at end of task N, never overwritten |
| Directory prefix | `YYYYMMDD_HHMMSS` — runs sort chronologically, nothing is overwritten |


**Step 2 — Start TensorBoard** (Terminal 2):
```bash
# Watch all runs at once (recommended):
tensorboard --logdir /absolute/path/to/runs/my_experiment

# Watch past runs (no current run needed)
tensorboard --logdir=runs

# Or watch a single specific run:
tensorboard --logdir /absolute/path/to/runs/my_experiment/20260617_143022_TBcartpole_single_2020
```

**Step 3 — Open browser:**
```
http://localhost:6006
```

Click **Scalars** to see training loss and reward curves. TensorBoard refreshes every 30 seconds, or use the ↺ button to reload manually.

**What is logged:**

| Tag | Tab in TensorBoard | Description |
| --- | --- | --- |
| `train/loss` | Scalars | MSE (mean squared error) loss of the dynamics model during training — lower is better |
| `train/learning_rate` | Scalars | Learning rate over time — useful to see when a scheduler reduces it |
| `train/weight_shift` | Scalars | How much the network weights changed since the last task — measures forgetting |
| `data/task_N/dataset_size` | Scalars | Number of collected transitions for task N — grows as the agent explores |
| `val/task_N/loss` | Scalars | Loss on the validation set for task N — how well the model predicts unseen data |
| `val/task_N/prediction_error` | Scalars | Average absolute prediction error on the validation set for task N |
| `train_env/task_N/reward` | Scalars | Total reward collected in one episode during training — higher is better |
| `train_env/task_N/episode_length` | Scalars | Number of steps the agent survived in one episode |
| `eval_env/task_N/reward` | Scalars | Mean reward over dedicated evaluation episodes — cleaner signal than training reward |
| `eval_env/task_N/prediction_error` | Scalars | How accurately the dynamics model predicts the next state during MPC rollouts — drops as the model improves |
| `eval_env/task_N/episode_time` | Scalars | Average time in seconds to complete one evaluation episode — reflects MPC planning cost |
| `train_env/task_N/koz_violations` | Scalars | Number of keep-out zone violations per episode — should go to zero as the agent learns |
| `train_env/task_N/theta_margin` | Scalars | Distance of the camera from the forbidden zone boundary every step — positive = safe, below -1/3 = violation |
| `train_env/task_N/attitude_error_deg` | Scalars | Attitude error in degrees every step — how far the satellite is from the target pointing direction, should decrease over training |
| `eval_env/task_N/koz_violations` | Scalars | Mean number of keep-out zone violations per evaluation episode — measures safety filter effectiveness in evaluation |
| `eval_env/task_N/state_mean` | Histograms | Mean (μ = average) of each state dimension across collected data |
| `eval_env/task_N/state_std` | Histograms | Std (standard deviation = how spread out values are) of each state dimension — grows if agent explores new states |
| `eval_env/task_N/action_mean` | Histograms | Mean (μ = average) of each action dimension across collected data |
| `eval_env/task_N/action_std` | Histograms | Std (standard deviation = how spread out values are) of each action dimension |
| `train/weights/*` | Histograms | Distribution of network weights — shows how the model changes over time |
| `eval_env/task_N/rollout_diff` | Images | Bar chart of open-loop prediction error per state dimension over the planning horizon |

> **Tip:** Always use `--savepath` with an absolute path in the `tensorboard --logdir` argument to avoid path confusion.

> **Tip:** `--play` and resume automatically find the most recent run directory for the given `--env`, `--method`, and `--seed` combination inside `--savepath`. If you have multiple runs and want to replay a specific one, pass its full timestamped path as `--savepath`.

## Safety filter

Every MPC action passes through a QP-based safety filter before reaching the environment:

```
u_proposed (MPC) → SafetyFilter.filter() → u_safe → env.step()
```

The filter minimises deviation from `u_proposed` subject to:
- **Hard constraint:** CBF barrier `H_dot ≥ ε` (KOZ never entered)
- **Soft constraint:** CLF decay `V_dot ≤ δ` (stability, can be relaxed)
- **Box constraint:** `u ∈ [−u_max, u_max]`

On solver failure it falls back to `u_proposed` and logs a warning.
The filter is geometry-agnostic — the env owns the CBF/CLF objects.

## Test
python -m pytest -s tests/test_cbf_clf_filter.py -v

## References

This project builds on the following works:

**Continual learning framework (HyperCRL):**
> Y. Huang, K. Xie, H. Bharadhwaj, and F. Shkurti, "Continual Model-Based
> Reinforcement Learning with Hypernetworks," *arXiv:2009.11997*, 2021.
> [arxiv.org/abs/2009.11997](https://arxiv.org/abs/2009.11997)

**Safety-guaranteed imitation learning from MPC (SafeMPC):**
> A. Meinert, N. Baldauf, P. Stadler, and A. Turnwald, "Safety-Guaranteed
> Imitation Learning from Nonlinear Model Predictive Control for Spacecraft
> Close Proximity Operations," *arXiv:2603.18910*, 2026.
> [arxiv.org/abs/2603.18910](https://arxiv.org/abs/2603.18910)

**Space environment and safety filter:**
> J. Yang and M. K. Ben-Larbi, "Safe Deep Reinforcement Learning for Spacecraft
> Reorientation with Pointing Keep-Out Constraint," in *Proc. CEAS EuroGNC 2026*,
> Madrid, Spain, May 2026, paper CEAS-GNC-2026-038.
> arXiv: [arxiv.org/abs/2605.19967](https://doi.org/10.48550/arXiv.2605.19967) [eess.SY]

