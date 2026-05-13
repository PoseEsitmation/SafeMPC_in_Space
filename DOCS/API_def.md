#
## Cross-subsystem interfaces

### `IAction` — Trajectory optimizer → Safety filter core

```python
u_proposed: np.ndarray  # shape (action_dim,)
```

The raw action from the MPC trajectory optimizer. Unchecked, potentially unsafe.

---

### `IObservation` — Gym state vector → Safety filter core

```python
state: np.ndarray  # shape (6,) = [p_x, p_y, p_z, v_x, v_y, v_z]
```

Must always expose position in `[:3]` and velocity in `[3:]`. CBF reads `state[:3]`, CLF reads the full vector.

---

### `ISafeAction` — Safety filter core → Gym step

```python
u_safe: np.ndarray  # shape (action_dim,)
```

The corrected action guaranteed to satisfy CBF and CLF constraints. This is the only action the environment ever receives.

---

### `ITransition` — Gym step → Replay buffer

```python
state:      np.ndarray  # shape (6,)
action:     np.ndarray  # shape (action_dim,) — u_safe, not u_proposed
next_state: np.ndarray  # shape (6,)
reward:     float
done:       bool
```

Note: the replay buffer stores `u_safe`, not `u_proposed`. The dynamics model trains on the same action distribution it will see at deployment.

---

### `ITaskConfig` — HyperCRL → Gym environment

```python
task_id:   int
u_max:     float        # actuator limit, passed to CBF constructor
goal:      np.ndarray   # shape (6,), passed to CLF constructor
obstacles: list[dict]   # [{"center": np.ndarray, "radius": float}]
```

Pushed at the start of each new task. The gym uses `task_id` to vary friction/gravity. The safety filter uses `u_max`, `goal`, and `obstacles` to re-instantiate CBF and CLF for the new task.

---

## Intra-subsystem interfaces — Safety filter

### `CBF.h(state)` → `float`

```python
def h(state: np.ndarray) -> float
# Returns h(x) = ||p − p_obs||² − r²
# Positive = safe, negative = inside KOZ
```

---

---

### `CBF.H_dot_expr(state, u_var)` → `cvxpy expression`

```python
def H_dot_expr(state: np.ndarray, u_var: cp.Variable) -> cp.Expression
# Returns a cvxpy expression linear in u_var
# Used as the CBF constraint inside the QP: H_dot_expr >= epsilon
```

---

### `CLF.V(state)` → `float`

```python
def V(state: np.ndarray) -> float
# Returns V(x) = (x − x_goal)ᵀ P (x − x_goal)
# Positive definite, zero only at goal
```

---

---

### `CLF.V_dot_expr(state, u_var)` → `cvxpy expression`

```python
def V_dot_expr(state: np.ndarray, u_var: cp.Variable) -> cp.Expression
# Returns a cvxpy expression linear in u_var
# Used as the CLF constraint inside the QP: V_dot_expr <= delta
```

---

### `SafetyFilter.filter(state, u_proposed)` → `np.ndarray`

```python
def filter(state: np.ndarray, u_proposed: np.ndarray) -> np.ndarray
# Core method. Builds and solves the QP.
# Returns u_safe on success, zeros on infeasibility.
```

---

### `IConstraintEvaluator.check(state, action)` → `bool`

```python
def check(state: np.ndarray, action: np.ndarray) -> bool
# Returns True if action satisfies CBF and CLF at given state
# Used by evaluator/logger and unit tests
```

---

## Intra-subsystem interfaces — HyperCRL

### `Hypernetwork.forward(task_id)` → `list[np.ndarray]`

```python
def forward(task_id: int) -> list[np.ndarray]
# Runs e_t → [hidden layers; θ] → [W1, b1, W2, b2, ...]
# Returns the full weight list for the main network
```

---

### `Hypernetwork.add_task(task_id)` → `None`

```python
def add_task(task_id: int) -> None
# Appends a new randomly-initialized embedding e_t
```

---

### `MainNetwork.forward(state_action, weights)` → `np.ndarray`

```python
def forward(state_action: np.ndarray, weights: list[np.ndarray]) -> np.ndarray
# Predicts next_state given injected weights
# shape in: (state_dim + action_dim,), shape out: (state_dim,)
```

---

### `MPCAgent.cache_hnet(task_id)` → `None`

```python
def cache_hnet(task_id: int) -> None
# Calls hnet.forward(task_id) and stores the weight list
# Called once per task before deployment
```

---

### `MPCAgent.act(state, task_id)` → `np.ndarray`

```python
def act(state: np.ndarray, task_id: int) -> np.ndarray
# Runs trajectory optimizer with cached weights
# Returns u_proposed — passes through safety_filter.filter() before env
```

---

## Intra-subsystem interfaces — Gym environment

### `Env.get_obstacles()` → `list[dict]`

```python
def get_obstacles() -> list[dict]
# Returns [{"center": np.ndarray, "radius": float}, ...]
# Called at task init to configure CBF
```

---

### `Env.step(action)` → `tuple`

```python
def step(action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]
# gymnasium API: (obs, reward, terminated, truncated, info)
# action received here is always u_safe
```

---

### `Env.reset(task_id)` → `np.ndarray`

```python
def reset(task_id: int) -> np.ndarray
# Resets environment for given task
# Returns initial state x_0
```
