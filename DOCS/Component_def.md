# Component Definition

[Figma board](https://www.figma.com/board/ZT3W5ffSM1kQXYwgdJTx4Q/Safety-Aware-HyperCRL-—-Implementation-Map?node-id=0-1&p=f&t=m6Esj7VUUtCnj1mH-0)

---

## Existing components (port to Python 3.12, no structural changes)

### Task embedding store
- one vector `e_t` per task, shape `(emb_dim,)`
- trained only on its own task via `emb_optimizer`

### Hypernetwork (`hnet`)
- MLP: `e_t → shared weights θ → [W1, b1, W2, b2, ...]`
- one embedding per task, shared `θ` across all tasks
- output: `list[torch.Tensor]`

### Main network (`mnet`)
- no trainable parameters — weights injected from `hnet`
- input: `xu (batch, state_dim + action_dim)` → output: `next_state (batch, state_dim)`
- dynamics model used in trajectory rollouts

### Hnet regularizer
- prevents forgetting: penalises drift from frozen snapshots of old task outputs
- `L_reg = Σ ‖hnet(e_j; θ) − snapshot_j‖²` for all `j < t`

### Target snapshots
- frozen outputs of `hnet` for tasks `0..t-1`, taken before training on task `t`
- never updated after snapshotting

### Dual optimizers
- `theta_optimizer` (Adam): updates shared `θ` with `L_task + β·L_reg`
- `emb_optimizer` (Adam): updates only current `e_t` with `L_task`

### Replay buffer (`DataCollector`)
- stores `(state, action, next_state)` — no reward, no done
- shapes: `(6,)`, `(3,)`, `(6,)` for spacecraft
- 75/25 train/val split at collection time, per task

### Training loop (`hnet_exp.py`)
- per task: snapshot → add embedding → iterate batches → update both optimizers

### MPC agent (`control/agent.py`)
- `cache_hnet(task_id)`: stores detached `list[Tensor]` weights once per task
- `act(state, task_id)`: runs trajectory optimizer, returns `u_proposed (3,)` as `torch.Tensor`

### Trajectory optimizer (CEM / MPPI / PDDM)
- owned by MPC agent as `self.control`
- calls `mnet` + `GTCost` repeatedly for rollouts
- outputs `u_proposed`

### Cost function (`GTCost`, `control/reward.py`)
- scores trajectories: `cost(x, u, t, task_id) → (batch,)`
- required input to CEM/MPPI/PDDM alongside `mnet`
- **needs a new `"spacecraft"` case**: quadratic tracking error + control effort

### Weight cache
- `self._cached_weights` on MPC agent
- avoids re-running `hnet` every step; filled once per task via `cache_hnet()`

### Environment handler (`CLEnvHandler`, `envs/cl_env.py`)
- creates and holds one env per task via `add_task(task_id)`
- drives the data collection loop: agent → env → replay buffer
- **needs `"spacecraft"` case added**

---

## New components (build from scratch)

### Spacecraft environment (`envs/spacecraft.py`)
- CWH orbital dynamics (eq. 1 in paper)
- state: `(6,)` = `[x, y, z, ẋ, ẏ, ż]` in LVLH frame
- action: `(3,)` = `[u₁, u₂, u₃]`, bounded `±0.082 m/s²`
- two phases: fly-around (KOZ active) and final approach (AC active)
- **owns CBF and CLF** — exposes `get_cbf()` and `get_clf()`
- swapping env automatically swaps the safety geometry

### `SphericalCBF`
- for fly-around phase; encodes keep-out zone as a sphere
- `h(x)`: positive outside KOZ, negative inside
- uses relative-degree-2 extension `H(x)` because `L_g h = 0` for CWH dynamics
- QP constraint is on `H_dot`, which is linear in `u`

### `ConicalCBF`
- for final-approach phase; encodes approach corridor as a cone
- `h(x)`: positive inside corridor, negative outside
- same relative-degree-2 extension as spherical
- only one CBF active at a time; env decides which to return from `get_cbf()`

### `QuadraticCLF`
- stability certificate; drives state toward decision point `x_goal`
- `V(x)`: quadratic, zero only at goal
- decay rate is state-dependent (fast near goal, negligible far away)
- `P` matrix: solved from CWH dynamics + cost matrices inside the env

### Safety filter (`control/safety_filter.py`)
- **env-independent**: constructed with `cbf` and `clf` from env
- single public method: `filter(state (6,), u_proposed (3,)) → u_safe (3,)`
- solves a small QP each timestep:
  - CBF constraint: hard (never relaxed)
  - CLF constraint: soft (slack allows it to yield to CBF)
  - input box constraint: `u in [-u_max, u_max]`
- insertion point in `agent.py`: one line between `act()` and `env.step()`

---

## Components to adapt

### `EnvSpecs` (`envs/cl_env.py`)
- add: `x_dims["spacecraft"] = 6`, `a_dims["spacecraft"] = 3`

### `GTCost` (`control/reward.py`)
- add `"spacecraft"` case: quadratic distance to goal + control effort penalty

### `CLEnvHandler` (`envs/cl_env.py`)
- add `"spacecraft"` branch in `add_task()`
- on task transition: call `env.get_cbf()` and `env.get_clf()` to rebuild filter

### Reward function
- soft obstacle penalty to reduce filter intervention rate during training

### Gym step / reset
- `step()`: returns 4-tuple `(obs (6,), reward, done, info)`
- `reset()`: returns `x_0 (6,)` — no args today; planned: accept `task_id`

### Evaluator / logger
- add: CBF violation count, filter intervention rate, reward with/without filter
