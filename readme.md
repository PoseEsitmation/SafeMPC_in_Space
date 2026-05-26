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

**Step 2 — Start TensorBoard** (Terminal 2):
```bash
tensorboard --logdir /absolute/path/to/runs/my_experiment
# e.g.: tensorboard --logdir /home/user/SafeMPC_in_Space/runs/cartpole
```

**Step 3 — Open browser:**
```
http://localhost:6006
```

Click **Scalars** to see training loss and reward curves. TensorBoard refreshes every 30 seconds, or use the ↺ button to reload manually.

**What is logged:**

| Tag | Tab in TensorBoard | Description |
| --- | --- | --- |
| `train/mse_loss` | Scalars | Dynamics model training loss |
| `train/regularizer` | Scalars | Weight shift between tasks |
| `val/task_N/loss` | Scalars | Validation loss per task |
| `val/task_N/diff` | Scalars | Validation prediction error per task |
| `rl/task_N/reward` | Scalars | Episode return during training |
| `rl/task_N/ep_len` | Scalars | Episode length during training |
| `rl/task_N/x_mu` | Histograms | Mean of state dimensions (normalization stats) |
| `rl/task_N/x_std` | Histograms | Std of state dimensions (normalization stats) |
| `rl/task_N/a_mu` | Histograms | Mean of action dimensions (normalization stats) |
| `rl/task_N/a_std` | Histograms | Std of action dimensions (normalization stats) |
| `train/weight/*` | Histograms | Network weight distributions |
| `task_N/rollout_avg_diff` | Images | Open-loop prediction error per state dimension |

> **Tip:** Always use `--savepath` with an absolute path in the `tensorboard --logdir` argument to avoid path confusion.
