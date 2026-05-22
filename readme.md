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
