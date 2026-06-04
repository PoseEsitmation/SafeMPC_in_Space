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
| `train/mse_loss` | Scalars | MSE (mean squared error) loss of the dynamics model during training — lower is better |
| `train/regularizer` | Scalars | How much the network weights changed since the last task — measures forgetting |
| `val/task_N/loss` | Scalars | Loss on the validation set for task N — how well the model predicts unseen data |
| `val/task_N/diff` | Scalars | Average absolute prediction error on the validation set for task N |
| `rl/task_N/reward` | Scalars | Total reward collected in one episode during training — higher is better |
| `rl/task_N/ep_len` | Scalars | Number of steps the agent survived in one episode |
| `rl/task_N/x_mu` | Histograms | Mean (μ = average) of each state dimension across collected data |
| `rl/task_N/x_std` | Histograms | Std (standard deviation = how spread out values are) of each state dimension — grows if agent explores new states |
| `rl/task_N/a_mu` | Histograms | Mean (μ = average) of each action dimension across collected data |
| `rl/task_N/a_std` | Histograms | Std (standard deviation = how spread out values are) of each action dimension |
| `train/weight/*` | Histograms | Distribution of network weights — shows how the model changes over time |
| `task_N/rollout_avg_diff` | Images | Bar chart of open-loop prediction error per state dimension over the planning horizon |

> **Tip:** Always use `--savepath` with an absolute path in the `tensorboard --logdir` argument to avoid path confusion.
