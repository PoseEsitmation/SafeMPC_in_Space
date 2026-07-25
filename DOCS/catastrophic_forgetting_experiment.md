# Catastrophic-Forgetting Experiment — Safety Filter in the Multi-Task Space Environment

**Goal.** Show that in a multi-task HyperCRL setting the CBF/CLF **safety filter is
(a) safe and (b) useful**, and that **DAgger distillation works**, *even while the
learned controller catastrophically forgets earlier tasks*.

The design turns catastrophic forgetting from a nuisance into the very thing that
demonstrates the filter's value: when the distilled policy forgets an old task's
avoidance behaviour, the non-learned filter is what keeps the satellite out of the
keep-out zone (KOZ).

---

## 1. Why this experiment is well-posed for *this* system

The pipeline contains two learners and one non-learner:

| Component | Learns? | Can forget? | Role |
|---|---|---|---|
| **Dynamics model** — `hnet` HyperCRL | yes, per-task hypernetwork weights | *designed not to* (task-conditioned + regularised) | feeds the MPC expert |
| **NN policy** — `PolicyNet`, distilled by BC + DAgger | yes — **one shared MLP, weights never reset** | **yes — naive continual learner** | the deployable distilled controller |
| **Safety filter** — CBF/CLF QP | **no** — geometry supplied by the env | **cannot forget** | the safety guarantee |

The key fact: [`PolicyTrainer.reset_per_task()`](../hypercrl/control/policy_net.py) clears
the DAgger buffer and the λ-curriculum **but not the policy weights**. So the distilled
policy is a textbook catastrophic-forgetting subject, while the QP filter is structurally
immune. That contrast is the experiment.

## 2. Hypotheses

- **H1 (SAFE).** Filtered KOZ violations stay ≈ 0 for *every* (after-task *k*, evaluated-task *j*),
  including old tasks re-tested after new learning. The guarantee survives forgetting.
- **H2 (FORGETTING IS REAL).** Unfiltered KOZ violations on an old task *j* **rise** as later
  tasks are learned — the raw policy overwrites its avoidance behaviour.
- **H3 (USEFUL).** `filter_saves = unfiltered_koz − filtered_koz` on old tasks **grows** with
  the number of tasks learned since — the filter actively prevents the catastrophe H2 creates.
- **H4 (DAgger WORKS MULTI-TASK).** Within each task, unfiltered KOZ and filter-intervention
  rate decline across DAgger rounds (already logged as `dagger_eval_*`); the forgetting matrix
  additionally shows whether that internalised safety **survives** later tasks or must be
  re-caught by the filter — the BC-only ablation isolates DAgger's contribution.

## 3. Design

- **Environment:** `spaceEnv` — 4 tasks that share dynamics but vary task *difficulty*
  (the axis that stresses the *policy*, i.e. the filter's target):

  | Task | Name | What varies |
  |---|---|---|
  | 0 | default | 80–180° init error, full torque (2 Nm), standard KOZ penalty |
  | 1 | easy | 10–45° init error |
  | 2 | hard | 90–180° init error + 5× KOZ penalty |
  | 3 | weak | half thruster power (0.5 Nm) |

  Each task's env carries **its own** CBF/CLF geometry; the forgetting eval re-attaches the
  correct per-task filter for every cell.

- **Conditions (2):**
  - `dagger` — full pipeline (BC + DAgger + CBF/CLF safety loss).
  - `bc` — BC-only ablation (`--no-dagger`, i.e. `dagger_every=0`): the policy still distils the
    MPC expert every dynamics update, but no learner-state DAgger rollouts are collected.
  - **Filter ON vs OFF is not a separate run** — every matrix cell evaluates the same frozen
    policy both filtered and unfiltered.

- **Seeds:** 3 (`0, 1, 2`) → mean ± std error bars on all forgetting metrics.

- **Profile:** the standard `spaceEnv` config (`default_arg_sat`): 4 tasks, 20 000 MPC
  steps/task, `policy_train_start=6000`, `dagger_every=500` (28 DAgger rounds/task),
  frozen normalisation. This is already the shortened "fast validation" profile
  (was 30 k), which keeps 3 seeds × 2 conditions tractable.

**Total: 6 training runs.**

## 4. Metrics — the forgetting matrix

After finishing task *k*, freeze the current policy and evaluate it on every task *j ≤ k*,
twice (filtered / unfiltered), for N episodes. This yields, per metric, a lower-triangular
matrix **R[k][j]**. Metrics captured per cell (both filters):

`koz_mean`, `koz_max`, `reward`, `min_margin_deg`, `worst_margin_deg`, `filter_frac`,
`fallback_frac`, `att_err_deg`.

Derived continual-learning metrics (per condition, over seeds):
- **ACC_final** = mean_j R[K][j] — final average performance across all tasks.
- **BWT** = mean_{j<K} (R[K][j] − R[j][j]) — backward transfer (forgetting sign).
- **Safety retention** = final filtered `koz_mean` (should be ≈ 0 → H1).
- **filter_saves(k)** = mean_{j<k} (unfiltered − filtered `koz_mean`) → H3.

Implementation: [`hnet_exp._eval_forgetting_matrix`](../hypercrl/hnet_exp.py) runs at every task
boundary and appends rows (crash-safe) to `forgetting_matrix.csv` in the run dir, mirroring to
TensorBoard under `forget/after_task_{k}/{filtered|unfiltered}/task_{j}/*`.

## 5. Protocol / commands

```bash
# All 6 runs (2 conditions x 3 seeds):
scripts/run_cf_experiment.sh cuda:0 0 1 2

# Or a single run manually:
python main.py run --method hnet --env spaceEnv --device cuda:0 --seed 0 \
    --savepath runs/cf --name cf_dagger --cf-experiment
python main.py run --method hnet --env spaceEnv --device cuda:0 --seed 0 \
    --savepath runs/cf --name cf_bc --cf-experiment --no-dagger

# Analyse -> figures + cl_metrics.csv:
python scripts/plot_forgetting.py --runs runs/cf --out runs/cf/analysis
```

New flags: `--cf-experiment` (enable the forgetting matrix), `--no-dagger` (BC-only ablation).
Knobs in `default_arg_policy`: `forget_eval_eps_filtered` (15), `forget_eval_eps_unfiltered` (40).

## 6. Expected results (what confirms each hypothesis)

| Figure | Reads as | Confirms |
|---|---|---|
| `matrix_koz.png` — filtered panel | ≈ 0 everywhere | **H1 safe** |
| `matrix_koz.png` — unfiltered panel | grows down each column (old tasks worsen) | **H2 forgets** |
| `retention_koz.png` | unfiltered task-0 KOZ rises with k; filtered flat at 0 | **H1 + H2** |
| `filter_saves.png` | rising curve; higher for `bc` than `dagger` | **H3 useful** |
| `matrix_reward.png` (filtered) | reward stays reasonable on old tasks | **useful** |
| `bc` vs `dagger` gap in unfiltered KOZ | DAgger's internalised safety decays slower | **H4** |
| `cl_metrics.png` | filtered final KOZ ≈ 0 for both; unfiltered high | **H1 vs H2 headline** |

## 7. Threats to validity / caveats

- **Frozen normaliser.** Norms are frozen after task 0's random phase (`freeze_norms=True`),
  so all tasks share one coordinate system — task differences must flow through the task
  embedding. The forgetting eval still re-caches per-task stats defensively.
- **Filter fallback.** When the raw policy drives into a hard-CBF-infeasible state the filter
  returns the *least-unsafe* action (soft-CBF fallback); `fallback_frac` is logged so filtered
  cells with residual KOZ hits are attributable, not silent.
- **Eval stochasticity.** Init attitude error and KOZ placement/size are randomised per reset;
  hence 40 unfiltered episodes/cell and ≥ 3 seeds. Use `--fixed-scenario` for a pinned-geometry
  variant if lower variance is needed.
- **Dynamics forgetting is not the axis.** `spaceEnv` tasks share dynamics, so this experiment
  isolates *policy* forgetting. To additionally stress the dynamics model / HyperCRL, re-run on
  `spaceEnv_moi` (inertia varies) — the harness is env-agnostic (`--env spaceEnv_moi`).

## 8. Compute budget

Per run: 4 tasks × 20 k MPC steps (CEM-dominated) plus the forgetting eval
(10 task-evals total across boundaries × 15 filtered QP episodes ≈ 150 QP-heavy episodes/run,
~tens of minutes; unfiltered episodes are nearly free). 6 runs total. Runnable on a single GPU;
the 3 seeds of each condition are independent and can be parallelised across GPUs.
