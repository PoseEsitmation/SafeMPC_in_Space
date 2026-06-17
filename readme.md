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
| `cartpole`          | Cartpole balancing                                                                               |
| `half_cheetah_body` | Half-cheetah with body variations                                                                |
| `pusher`            | Pusher manipulation task                                                                         |
| `door_pose`         | Door opening task                                                                                |
| `spaceEnv`          | Satellite attitude control with KOZ — initial error 80°–180°, standard penalty (β=10, α=66)     |
| `spaceEnv_easy`     | Same as `spaceEnv` but small initial error (10°–45°) — easier slew task                         |
| `spaceEnv_hard`     | Large initial error (90°–180°) + 5× stronger KOZ penalty (β=50, α=100) — harder constraint task |
| `spaceEnv_weak`     | Half thruster power (0.5 Nm) — simulates a low-torque spacecraft                                |

> **Tip:** The space environment can be configured by these parameters via constructer arguments.

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

`model.pt` is overwritten every `save_every` env steps (default 1000) so a crash never loses more than 1000 steps of progress. Per-task snapshots (`model_N.pt`) are written once at the end of each task and never overwritten.

The timestamp prefix (`YYYYMMDD_HHMMSS`) ensures runs sort chronologically on disk and in the TensorBoard directory picker. Every run is kept independently — nothing is overwritten.

**Step 2 — Start TensorBoard** (Terminal 2):
```bash
# Watch all runs at once (recommended):
tensorboard --logdir /absolute/path/to/runs/my_experiment

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
