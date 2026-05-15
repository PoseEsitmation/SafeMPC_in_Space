# API Definition

---

## Data flow overview

```
Env ──get_cbf()/get_clf()──► SafetyFilter
                                  ▲
Hypernetwork ──W_t──► mnet        │
                        │         │
                        ▼         │
              Trajectory optimizer │
                        │         │
                        ▼         │
                   u_proposed ────┘
                        │
                   SafetyFilter.filter()
                        │
                        ▼
                     u_safe ──► Env.step()
                                    │
                                    ▼
                             Replay buffer (s, a, s')
```

**Key insertion point:** `u_proposed` → `SafetyFilter.filter()` → `u_safe`. One line in `agent.py`.

---

## Cross-subsystem interfaces

### `IAction` — optimizer → safety filter
- `u_proposed: torch.Tensor` — shape `(3,)` for spacecraft
- `agent.act()` returns a Tensor; caller calls `.detach().cpu().numpy()` first

### `ISafeAction` — safety filter → env
- `u_safe: np.ndarray` — shape `(3,)`
- only action env ever receives

### `ITransition` — env → replay buffer
- `state: np.ndarray` — shape `(6,)`
- `action: np.ndarray` — shape `(3,)`, always `u_safe`
- `next_state: np.ndarray` — shape `(6,)`
- no reward, no done flag

### `ITaskConfig` — training loop → env
- `task_id: int`
- env derives all geometry from it; caller passes nothing else

---

## Safety filter interfaces

> Constructed once per task: `SafetyFilter(cbf=env.get_cbf(), clf=env.get_clf())`  
> Filter has zero knowledge of geometry — it only calls methods on the objects it received.

### `CBF` (lives in env, passed to filter)
- `CBF.H(state: np.ndarray) -> float`
  - state shape: `(6,)`
  - relative-degree-2 barrier; used in QP, not `h(x)` directly
- `CBF.H_dot_expr(state: np.ndarray, u_var: cp.Variable) -> cp.Expression`
  - linear in `u_var`
  - QP constraint: `H_dot_expr >= epsilon`

### `CLF` (lives in env, passed to filter)
- `CLF.V(state: np.ndarray) -> float`
  - state shape: `(6,)`
- `CLF.V_dot_expr(state: np.ndarray, u_var: cp.Variable) -> cp.Expression`
  - linear in `u_var`
  - QP constraint: `V_dot_expr <= delta`

### `SafetyFilter.filter(state, u_proposed) -> np.ndarray`
- inputs: `state (6,)`, `u_proposed (3,)`
- output: `u_safe (3,)`
- solves QP each step:
  - minimize deviation from `u_proposed`
  - CBF constraint: hard
  - CLF constraint: soft (slack `delta`)
  - input box: `u in [-u_max, u_max]`
- on solver failure: returns `u_proposed`, logs fallback

### `IConstraintEvaluator.check(state, action) -> bool`
- state: `(6,)`, action: `(3,)`
- used by logger and unit tests

---

## HyperCRL interfaces

### `Hypernetwork.forward(task_id: int) -> list[torch.Tensor]`
- input: task id
- output: weight list `[W1, b1, W2, b2, ...]` as tensors

### `Hypernetwork.add_task(task_id: int, std_normal_temb: float) -> None`
- appends new embedding `e_t`

### `MainNetwork.forward(xu: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor`
- `xu` shape: `(batch, state_dim + action_dim)` = `(batch, 9)`
- output shape: `(batch, 6)` — or `(batch, 12)` if `out_var=True`

### `MPCAgent.cache_hnet(task_id: int) -> None`
- stores detached weights in `self._cached_weights`
- call once per task before rollout

### `MPCAgent.act(state, task_id=None, first_action=True) -> torch.Tensor`
- output shape: `(3,)` — `u_proposed`

### `GTCost.__call__(x, u, t, task_id) -> torch.Tensor`
- `x` shape: `(batch, 6)`, `u` shape: `(batch, 3)`
- output: `(batch,)` cost per trajectory sample
- needs a `"spacecraft"` case added

---

## Env interfaces

> Each env owns its CBF and CLF. Swapping env = swapping safety geometry automatically.

### `Env.get_cbf() -> CBF`
- returns the active CBF for the current phase
- spacecraft: `SphericalCBF` (fly-around) or `ConicalCBF` (final approach)

### `Env.get_clf() -> CLF`
- constructs and returns a `QuadraticCLF`
- solves DARE internally; caller gets a ready-to-use object

### `Env.step(action: np.ndarray) -> tuple`
- input: `u_safe (3,)`
- output: `(obs (6,), reward, done, info)` — old gym 4-tuple

### `Env.reset() -> np.ndarray`
- output: `x_0 (6,)`
- planned: accept `task_id`

### `CLEnvHandler.add_task(task_id: int) -> Env`
- creates and registers env for the given task
- called at start of each task in the training loop
