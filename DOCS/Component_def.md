
## Existing HyperCRL components

> Port to Python 3.12. No structural changes required.

### Task embedding store

Holds one learned embedding vector `e_t` per task. These are small dense vectors that encode task identity. The hypernetwork reads them to generate task-specific weights. Each embedding is trained only on its own task's data via `emb_optimizer`.

---

### Hypernetwork (`hnet`)

The core of HyperCRL. A standard MLP whose output is the full weight vector for the main network. It has shared parameters `θ` across all tasks, plus one embedding per task. When you call `hnet.forward(task_id=t)`, it runs `e_t → [hidden layers] → [W1, b1, W2, b2, ...]`.

---

### Main network (`mnet`)

A weight-less MLP shell. It has no trainable parameters of its own. It takes `(state, action)` as input and predicts `next_state`, but only works when weights are injected from the hypernetwork. This is the learned dynamics model used inside MPC rollouts.

---

### Hnet regularizer

Computes the anti-forgetting loss `L_reg = Σ ||hnet(e_j; θ) − target_j||²` for all previous tasks `j < t`. If this term is zero, the hypernetwork still produces the exact same weights for old tasks as it did before training on the new one — meaning the dynamics model for old tasks is unchanged.

---

### Target snapshots

Before training on task `t`, the system freezes `hnet(e_0), hnet(e_1), ..., hnet(e_{t-1})` as fixed targets. The regularizer measures drift from these snapshots. They are never updated after being snapshotted.

---

### Dual optimizers

Two separate Adam optimizers run in parallel. `θ_optimizer` updates the shared hypernetwork weights using both `L_task` and `L_reg`. `emb_optimizer` updates only the current task's embedding `e_t` using `L_task` alone, so old embeddings are never touched.

---

### Replay buffer

Stores `(state, action, next_state)` tuples collected by the MPC agent interacting with the environment. The training loop samples batches from this to compute `L_task`. Each task has its own buffer.

---

### Training loop (`hnet_exp.py`)

The main outer loop. For each task: snapshot targets, add embedding, then iterate: sample batch → compute `L_task + β·L_reg` → update both optimizers. Continues until convergence, then moves to the next task.

---

### MPC agent (`control/agent.py`)

Calls `cache_hnet(task_id)` once to store the weight list, then at each timestep calls the trajectory optimizer to plan ahead using the cached dynamics model. Returns the first action of the best trajectory found.

---

### Trajectory optimizer

Runs CEM, MPPI, or PDDM to find the action sequence that maximizes reward over a horizon. Uses `mnet.forward(state, action; W_t)` repeatedly for rollouts. Outputs `u_proposed` — the raw, unchecked action.

---

### Weight cache

A simple dictionary mapping `task_id → weight list`. Avoids re-running the hypernetwork forward pass at every MPC timestep. Populated once per task before deployment.

---

## Components to implement

> Build from scratch. None of these exist in the current codebase.

### CBF` — Control Barrier Function

The mathematical safety certificate. Defines a forbidden zone as a sphere around an obstacle. The core function is:


---

### `QuadraticCLF` — Control Lyapunov Function

The stability certificate. Ensures the robot converges to the goal rather than just bouncing around safely.

---

### Safety filter core (`control/safety_filter.py`)

The main class that wires CBF and CLF together. Exposes a single method `filter(state, u_proposed) → u_safe`. This is the one-line insertion point into `agent.py`. Internally builds and solves the following QP at every timestep:

If the  returns `optimal` or `optimal_inaccurate`, return `u_sf.value`. Otherwise hand off to the fallback handler.

---

---

## Components to adapt

> Exist in the current codebase but require modification.

### State vector

The gym observation must expose a flat array `[p_x, p_y, p_z, v_x, v_y, v_z]` in a known, fixed order. The CBF constructor reads position `p = state[:3]` and the CLF reads the full state vector. If HyperCRL's existing environments do not expose this cleanly, wrap the observation with a custom `ObservationWrapper`.

---

### Obstacle registry

The environment must expose obstacle positions and radii so the CBF can be instantiated with the right parameters. Add a method `env.get_obstacles() → list[{"center": np.ndarray, "radius": float}]` that the safety filter setup code can query at initialization.

---

### Reward function

No structural change needed, but add a soft obstacle penalty term so the MPC agent learns to prefer safe actions during training, reducing how often the filter needs to intervene at deployment.

---

### Task manager

HyperCRL already varies task parameters (friction, gravity). Ensure the task config change also updates the CBF's `u_max` and the CLF's `P` matrix if the dynamics change significantly between tasks.


---

### Gym step / reset


---

### Evaluator / logger

Extend with three new metrics: CBF violation count per episode (should be zero), safety filter intervention rate (fraction of steps where `u_safe ≠ u_proposed`), and the performance vs safety tradeoff plot comparing reward with and without the filter.
