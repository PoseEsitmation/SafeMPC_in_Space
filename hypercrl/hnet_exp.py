import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import math
import torch
import os

from torch.utils.data import DataLoader

from hypercrl.tools import reset_seed, str_to_act
from hypercrl.tools import MonitorHnet, HP, Hparams
from hypercrl.control import RandomAgent, MPC, SafeAgent, SafetyFilter
from hypercrl.control.agent import NNPolicyAgent
from hypercrl.control.policy_net import (PolicyNet, PolicyTrainer,
                                          make_cheetah_cbf_fn, make_space_cbf_fn,
                                          make_space_clf_fn, make_space_margin_fn,
                                          make_space_boundary_sampler,
                                          make_space_cbf_feasible_fn)
from hypercrl.control.agent import _preprocess_state_torch
from hypercrl.envs.cl_env import CLEnvHandler
from hypercrl.dataset.datautil import DataCollector

from hypercrl.model import build_model_hnet as build_model
from hypercrl.model import reload_model_hnet as reload_model
from hypercrl.hypercl.utils import hnet_regularizer as hreg
from hypercrl.hypercl.utils import ewc_regularizer as ewc
from hypercrl.hypercl.utils import si_regularizer as si
from hypercrl.hypercl.utils import optim_step as opstep

class TaskLoss(torch.nn.Module):
    def __init__(self, hparams, mnet):
        super(TaskLoss, self).__init__()
        self.out_var = hparams.out_var
        self.y_dim = hparams.out_dim
        self.mlp_var_minmax = hparams.mlp_var_minmax

        self.reg_norm = 1
        self.reg_lambda = hparams.reg_lambda
        self.mnet = mnet

    def regularize(self, weights):
        if self.reg_lambda == 0:
            return 0

        loss = 0
        for weight in weights:
            loss += weight.norm(self.reg_norm)

        # Scale by reg_lambda — the unweighted sum of L1 norms over all
        # generated mnet tensors is O(100), which dwarfs the task MSE and
        # lasso-crushes the generated weights: the dynamics model plateaued
        # at ~50% unexplained variance in runs baseline_11..19 (train/loss
        # pinned at ~103) while the same data was fittable to ~0.04 L1.
        # (The pre-fix code accidentally kept only the LAST tensor's norm,
        # i.e. effectively no regularization — and the model fit to 0.02.)
        return self.reg_lambda * loss

    def reg_logvar(self, weights):
        if self.mlp_var_minmax:
            max_logvar = self.mnet.mlp_max_logvar
            min_logvar = self.mnet.mlp_min_logvar
        else:
            max_logvar = weights[0]
            min_logvar = weights[1]

        # Regularize max/min var
        loss = 0.01 * (max_logvar.sum() - min_logvar.sum())
        return loss

    def forward(self, pred, gt, weights, add_reg_logvar=True):
        if self.out_var:
            mu, logvar = torch.split(pred, self.y_dim, dim=-1)

            # Compute loss of a task (i.e during evaluation)
            inv_var = torch.exp(-logvar)
            loss = ((mu - gt) ** 2) * inv_var + logvar
            loss = loss.sum() / self.y_dim
           
            if add_reg_logvar:
                loss += self.reg_logvar(weights)

        else:
            loss = torch.nn.functional.mse_loss(pred, gt, reduction='sum')
            loss = loss / self.y_dim

        loss += self.regularize(weights)

        return loss

class TaskLossMT(TaskLoss):
    def __init__(self, hparams, mnet, hnet, collector, task_id):
        super().__init__(hparams, mnet)
        self.hnet = hnet
        self.task_id = task_id
        self.hparams = hparams
        self.device = hparams.device
    
        self.add_trainset(collector, task_id, hparams)

    def add_trainset(self, collector, task_id, hparams):
        old_data, old_data_iter = [], []
        for tid in range(0, task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.bs,
                shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter

    def replay(self, tid):
        loader_it = self.old_data_iter[tid]
        try:
            data = next(loader_it)
        except StopIteration:
            # Reset the dataloader iterable
            loader_it = iter(self.old_data[tid])
            self.old_data_iter[tid] = loader_it
            data = next(loader_it)

        x_t, a_t, x_tt = data
        x_t, a_t, x_tt = x_t.to(self.device), a_t.to(self.device), x_tt.to(self.device)

        # Forward Pass
        X = torch.cat((x_t, a_t), dim=-1)
        weights = self.hnet.forward(tid)
        Y = self.mnet.forward(X, weights)
        
        # Task-specific loss.
        loss_task = super().forward(Y, x_tt, weights, add_reg_logvar=False)
        return loss_task

    def forward(self, pred, gt, weights):
        loss = super().forward(pred, gt, weights)
        for tid in range(0, self.task_id):
            loss += self.replay(tid)
        return loss

class TaskLossReplay(TaskLossMT):
    def __init__(self, hparams, mnet, hnet, collector, task_id):
        super().__init__(hparams, mnet, hnet, collector, task_id)
    
    def add_trainset(self, collector, task_id, hparams):
        old_data, old_data_iter = [], []
        M = hparams.bs // task_id if task_id > 0 else 0
        for tid in range(0, task_id):
            train_set, _ = collector.get_dataset(tid)

            train_loader = torch.utils.data.DataLoader(train_set, batch_size=M,
                shuffle=True, drop_last=True)
            old_data.append(train_loader)
            old_data_iter.append(iter(train_loader))

        self.old_data = old_data
        self.old_data_iter = old_data_iter
    

def augment_model(task_id, mnet, hnet, collector, hparams):

    # Regularizer targets.
    targets = hreg.get_current_targets(task_id, hnet)

    # Add new hypernet embeddings and Loss Function
    hnet.add_task(task_id, hparams.std_normal_temb)

    if hparams.model == "hnet_mt":
        # Loss Function
        mll = TaskLossMT(hparams, mnet, hnet, collector, task_id)
    elif hparams.model == "hnet_replay":
        mll = TaskLossReplay(hparams, mnet, hnet, collector, task_id)
    else:
        mll = TaskLoss(hparams, mnet)

    # (Re)Put model to GPU
    device = hparams.device
    mnet.to(device)
    hnet.to(device)

    # Optimize over the GP model params and likelihood param
    
    mnet.train()
    hnet.train()

    # Collect Fisher estimates for the reg computation.
    fisher_ests = None
    if hparams.ewc_weight_importance and task_id > 0:
        fisher_ests = []
        n_W = len(hnet.target_shapes)
        for t in range(task_id):
            ff = []
            for i in range(n_W):
                _, buff_f_name = ewc._ewc_buffer_names(t, i, False)
                ff.append(getattr(mnet, buff_f_name))
            fisher_ests.append(ff)

    # Register SI buffers for new task
    si_omega = None
    if hparams.model == "hnet_si":
        si.si_register_buffer(mnet, hnet, task_id)
        if task_id > 0:
            si_omega = si.get_si_omega(mnet, task_id)

    regularized_params = list(hnet.theta)
    if task_id > 0 and hparams.plastic_prev_tembs:
        for i in range(task_id): # for all previous task embeddings
            regularized_params.append(hnet.get_task_emb(i))
    theta_optimizer = torch.optim.Adam(regularized_params, lr=hparams.lr_hyper)
    # We only optimize the task embedding corresponding to the current task,
    # the remaining ones stay constant.
    emb_optimizer = torch.optim.Adam([hnet.get_task_emb(task_id)],
                               lr=hparams.lr_hyper)

    trainer_misc = (targets, mll, theta_optimizer, emb_optimizer,regularized_params,
        fisher_ests, si_omega)

    return trainer_misc

def augment_model_after(task_id, mnet, hnet, hparams, collector):
    if hparams.model == "hnet_si":
        si.update_omega(mnet, hnet, hparams.si_eps, task_id)

    if hparams.ewc_weight_importance:
        ## Estimate Fisher for outputs of the hypernetwork.
        weights = hnet.forward(task_id)

        # Note, there are actually no parameters in the main network.
        fake_main_params = torch.nn.ParameterList()
        for i, W in enumerate(weights):
            fake_main_params.append(torch.nn.Parameter(torch.Tensor(*W.shape),
                                                 requires_grad=True))
            fake_main_params[i].data = weights[i].detach().to(hparams.device)

        ewc.compute_fisher(task_id, collector, fake_main_params, hparams.device, mnet,
            empirical_fisher=True, online=False, n_max=hparams.n_fisher,
            regression=True, allowed_outputs=None, out_var=hparams.out_var)

def train(task_id, mnet, hnet, trainer_misc, logger, train_set, hparams):

    # Data Loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=hparams.bs, shuffle=True,
            drop_last=True, num_workers=hparams.num_ds_worker)

    # DEVICE
    device = hparams.device

    regged_outputs = None

    targets, mll, theta_optimizer, emb_optimizer,regularized_params, \
        fisher_ests, si_omega = trainer_misc

    # Whether the regularizer will be computed during training?
    calc_reg = task_id > 0 and hparams.beta > 0

    it = 0
    while it < hparams.train_dynamic_iters:
        mnet.train()
        hnet.train()
        for i, data in enumerate(train_loader):
            if len(data) == 3:
                x_t, a_t, x_tt = data
                x_t, a_t, x_tt = x_t.to(device), a_t.to(device), x_tt.to(device)
                X = torch.cat((x_t, a_t), dim=-1)
            else:
                X, x_tt = data
                X, x_tt = X.to(device), x_tt.to(device)

            ### Train theta and task embedding.
            theta_optimizer.zero_grad()
            emb_optimizer.zero_grad()

            weights = hnet.forward(task_id)
            if hparams.model == "hnet_si":
                si.si_update_optim_step(mnet, weights, task_id)
                for weight in weights:
                    weight.retain_grad() # save grad for calculate si path integral

            Y = mnet.forward(X, weights)
            # Task-specific loss.
            loss_task = mll(Y, x_tt, weights)
            # We already compute the gradients, to then be able to compute delta
            # theta.
            loss_task.backward(retain_graph=calc_reg,
                            create_graph=hparams.backprop_dt and calc_reg)
            torch.nn.utils.clip_grad_norm_(hnet.get_task_emb(task_id), hparams.grad_max_norm)

            # The task embedding is only trained on the task-specific loss.
            # Note, the gradients accumulated so far are from "loss_task".
            emb_optimizer.step()

            # SI
            if hparams.model == "hnet_si":
                torch.nn.utils.clip_grad_norm_(weights, hparams.grad_max_norm)
                si.si_update_grad(mnet, weights, task_id)

            # Update Regularization
            loss_reg = torch.tensor(0., requires_grad=False)
            dTheta = None
            grad_tloss = None
            if calc_reg:
                if i % 1000 == 0:  # Just for debugging: displaying grad magnitude.
                    grad_tloss = torch.cat([d.grad.clone().view(-1) for d in
                                            hnet.theta])
                if hparams.no_look_ahead:
                    dTheta = None
                else:
                    dTheta = opstep.calc_delta_theta(theta_optimizer,
                        hparams.use_sgd_change, lr=hparams.lr_hyper,
                        detach_dt=not hparams.backprop_dt)

                if hparams.plastic_prev_tembs:
                    dTembs = dTheta[-task_id:]
                    dTheta = dTheta[:-task_id] if dTheta is not None else None
                else:
                    dTembs = None

                loss_reg = hreg.calc_fix_target_reg(hnet, task_id,
                    targets=targets, dTheta=dTheta, dTembs=dTembs, mnet=mnet,
                    inds_of_out_heads=regged_outputs,
                    fisher_estimates=fisher_ests,
                    si_omega=si_omega)
                
                loss_reg = loss_reg * hparams.beta * Y.size(0)

                loss_reg.backward()

                if grad_tloss is not None: # Debug
                    grad_full = torch.cat([d.grad.view(-1) for d in hnet.theta])
                    # Grad of regularizer.
                    grad_diff = grad_full - grad_tloss
                    grad_diff_norm = torch.norm(grad_diff, 2)
                    
                    # Cosine between regularizer gradient and task-specific
                    # gradient.
                    if dTheta is None:
                        dTheta = opstep.calc_delta_theta(theta_optimizer,
                            hparams.use_sgd_change, lr=hparams.lr_hyper,
                            detach_dt=not hparams.backprop_dt)
                    dT_vec = torch.cat([d.view(-1).clone() for d in dTheta])
                    grad_cos = torch.nn.functional.cosine_similarity(grad_diff.view(1,-1),
                                                dT_vec.view(1,-1))

                    grad_tloss = (grad_tloss, grad_full, grad_diff_norm, grad_cos)

            torch.nn.utils.clip_grad_norm_(regularized_params, hparams.grad_max_norm)
            theta_optimizer.step()

            logger.train_step(loss_task, loss_reg, dTheta, grad_tloss, weights)       
            # Validate
            logger.validate(mll)

            it += 1
            if it >= hparams.train_dynamic_iters:
                break

def plot_embs(hparams, embs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for emb in embs:
        emb = emb.detach().cpu().numpy()
        ax.plot(emb[0], emb[1], 'kx')
    fig.savefig(f'{hparams.save_folder}/embedding_{hparams.seed}.png')

def play_model(hparams):
    _, hnet, agent, checkpoint, _ = reload_model(hparams, need_data=False)

    # Reset seed
    reset_seed(hparams.seed)

    # Task Embedding
    embs = hnet.get_task_embs()
    #plot_embs(hparams, embs)

    num_tasks = checkpoint['num_tasks_seen']
    if num_tasks == 0:
        print("[play] No completed task checkpoint found — playing current model.pt")
        num_tasks = 1

    envs = CLEnvHandler(hparams.env, hparams.seed)
    for task_id in range(num_tasks):
        # Cache the mainnet weight
        agent.cache_hnet(task_id)
        env = envs.add_task(task_id, render=True)

        avg_rewards = []
        for _ in range(10):
            rewards = []
            x_t, _ = env.reset()
            agent.reset()
            done = False
            while (not done):
                env.render()
                u_t = agent.act(x_t, task_id=task_id).cpu().numpy()
                x_tt, reward, done, _ = env.step(u_t.reshape(env.action_space.shape))
                x_t = x_tt
                rewards.append(reward)
            eprew = np.sum(rewards)
            avg_rewards.append(eprew)
            print(f"Task {task_id + 1}, episode reward {eprew}, ep length {len(rewards)}")

        avg_reward = np.mean(avg_rewards)
        print(f"Average reward for task {task_id + 1} is {avg_reward}")

def _eval_nn_policy(
    nn_agent, env, task_id: int, writer, global_step: int,
    n_episodes: int = 1,
    tag_prefix: str = "policy_eval",
    disable_filter: bool = False,
) -> None:
    """Evaluate the NN policy for n_episodes and log diagnostics to TensorBoard.

    Key metrics
    -----------
    filter_fraction  — fraction of steps where the safety filter corrected the
        NN action.  Primary indicator that DAGGER is working: should trend → 0.
    filter_du_mean   — mean ‖u_safe − u_nn‖ (correction magnitude).
    filter_du_max    — worst-case correction in the episode.
    reward           — episode return.
    koz_violations   — keep-out zone violations (should stay 0 with filter active).
    min_theta_margin_deg — worst (smallest) KOZ margin seen, in degrees.
    att_err_final_deg    — attitude error at the end of the episode, in degrees.

    Parameters
    ----------
    tag_prefix : str
        TensorBoard tag namespace, e.g. "policy_eval" or "dagger_eval_unfiltered".
    disable_filter : bool
        If True, temporarily detach the agent's safety filter for the duration
        of this eval (raw NN output applied directly) — used to check whether
        the *learned policy itself* is safe, independent of the QP filter.
        The filter is restored afterwards regardless of outcome.
    """
    all_rewards       = []
    all_koz           = []
    all_filter_frac   = []
    all_fallback_frac = []
    all_du_mean       = []
    all_du_max        = []
    all_min_margin    = []
    all_att_err       = []

    saved_filter = getattr(nn_agent, "safety_filter", None)
    if disable_filter:
        nn_agent.set_safety_filter(None)
    sf = getattr(nn_agent, "safety_filter", None)

    try:
        for _ in range(n_episodes):
            x_t, _ = env.reset()
            nn_agent.reset()
            done         = False
            ep_reward    = 0.0
            koz_hits     = 0
            n_steps      = 0
            n_filter     = 0
            n_fallback   = 0
            du_norms     = []
            min_margin   = float("inf")
            att_err_final = float("nan")

            while not done:
                u = nn_agent.act(x_t, task_id=task_id).detach().cpu().numpy()
                x_tt, reward, terminated, truncated, info = env.step(
                    u.reshape(env.action_space.shape))
                done       = terminated or truncated
                ep_reward += reward
                n_steps   += 1
                if info.get("keep_out_violation"):
                    koz_hits += 1
                if sf is not None:
                    du_norms.append(float(sf.last_du_norm))
                    if sf.last_was_active:
                        n_filter += 1
                    # Fallback = hard CBF infeasible for the NN's action; the
                    # filter could only return the least-unsafe correction, so
                    # this counts steps where safety was not guaranteed.
                    if getattr(sf, "last_type", 0) == SafetyFilter.TYPE_FALLBACK:
                        n_fallback += 1
                theta_margin_deg = np.degrees(
                    (x_tt[7] + 1.0) * (3 * np.pi / 4) - np.pi / 2)
                min_margin = min(min_margin, float(theta_margin_deg))
                att_err_final = 2 * np.degrees(
                    np.arccos(np.clip(np.abs(x_tt[0]), 0.0, 1.0)))
                x_t = x_tt

            all_rewards.append(ep_reward)
            all_koz.append(koz_hits)
            all_filter_frac.append(n_filter / max(n_steps, 1))
            all_fallback_frac.append(n_fallback / max(n_steps, 1))
            all_min_margin.append(min_margin)
            all_att_err.append(float(att_err_final))
            if du_norms:
                all_du_mean.append(float(np.mean(du_norms)))
                all_du_max.append(float(np.max(du_norms)))
    finally:
        if disable_filter:
            nn_agent.set_safety_filter(saved_filter)

    avg_reward     = float(np.mean(all_rewards))
    avg_koz        = float(np.mean(all_koz))
    avg_filt_frac  = float(np.mean(all_filter_frac))
    avg_fb_frac    = float(np.mean(all_fallback_frac))
    avg_du_mean    = float(np.mean(all_du_mean)) if all_du_mean else 0.0
    avg_du_max     = float(np.max(all_du_max))   if all_du_max  else 0.0
    avg_min_margin = float(np.mean(all_min_margin))
    avg_att_err    = float(np.nanmean(all_att_err))
    # Worst case across episodes — with many validation episodes the means
    # above hide the tail, and the tail is what safety is about.
    worst_min_margin = float(np.min(all_min_margin))
    max_koz          = float(np.max(all_koz))

    writer.add_scalar(f"{tag_prefix}/task_{task_id}/reward",               avg_reward,     global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/koz_violations",       avg_koz,        global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/filter_fraction",      avg_filt_frac,  global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/filter_fallback_frac", avg_fb_frac,    global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/filter_du_mean",       avg_du_mean,    global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/filter_du_max",        avg_du_max,     global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/min_theta_margin_deg", avg_min_margin, global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/att_err_final_deg",    avg_att_err,    global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/min_theta_margin_worst", worst_min_margin, global_step)
    writer.add_scalar(f"{tag_prefix}/task_{task_id}/koz_violations_max",     max_koz,          global_step)
    print(
        f"  [{tag_prefix}] task {task_id}  n_eps={n_episodes}  reward={avg_reward:.2f}"
        f"  koz={avg_koz:.1f} (max {max_koz:.0f})"
        f"  min_margin={avg_min_margin:.2f}deg (worst {worst_min_margin:.2f})"
        f"  filter_frac={avg_filt_frac:.3f}  fallback_frac={avg_fb_frac:.3f}"
        f"  du_mean={avg_du_mean:.3f}  du_max={avg_du_max:.3f}"
    )
    return {"koz": avg_koz, "reward": avg_reward, "filter_frac": avg_filt_frac}


def _log_filter_success(writer, task_id: int, step: int, res_f: dict, res_u: dict) -> None:
    """Filter success: violations the QP prevented for the SAME policy.

    The filtered and unfiltered validations run the identical policy at the
    identical training step (on fresh random episodes), so the difference in
    KOZ violations is attributable to the filter:

        saves_per_ep  = unfiltered_koz − filtered_koz
        success_rate  = 1 − filtered_koz / unfiltered_koz

    When the raw policy no longer violates at all (unfiltered_koz = 0) there
    is nothing left to save — the rate is logged as 1.0 (vacuously perfect,
    and the desired end state).
    """
    if writer is None or res_f is None or res_u is None:
        return
    saves = max(res_u["koz"] - res_f["koz"], 0.0)
    if res_u["koz"] > 0:
        rate = min(max(1.0 - res_f["koz"] / res_u["koz"], 0.0), 1.0)
    else:
        rate = 1.0
    writer.add_scalar(f"dagger_eval/task_{task_id}/filter_saves_per_ep",  saves, step)
    writer.add_scalar(f"dagger_eval/task_{task_id}/filter_success_rate", rate,  step)
    print(f"  [filter] prevented {saves:.1f} violation-steps/ep  "
          f"success_rate={rate:.0%}")


def run(hparams):

    print("[DEBUG run] ENTRY:", hparams.device)

    # Fixed-scenario mode (paper-equivalent evaluation difficulty): pin the
    # per-episode randomisation to one corridor geometry — init attitude
    # error 120–140° and a constant 20° KOZ half-angle (the KOZ centre is
    # already deterministic: ratio 0.5 midway between start and goal
    # boresight).  This mirrors the paper's single fixed CR scenario, where
    # raw-policy avoidance is learnable; the fully randomised default asks
    # the network to generalise planner-like across arbitrary geometries.
    # SatDynEnv.reset() reads these module globals at call time.
    if getattr(hparams, "space_fixed_scenario", False) \
            and hparams.env.startswith("spaceEnv"):
        import hypercrl.envs.space_KOZ as _sk
        _sk.angle_bound_lower  = 120
        _sk.angle_bound_upper  = 140
        _sk.half_angle_low_deg  = 20.0
        _sk.half_angle_high_deg = 20.0
        # Offset the cone centre 10° off the start→goal arc: still blocks the
        # direct path (half-angle 20°) but one detour side is clearly shorter,
        # so the expert's avoidance is unimodal — plain MSE imitation of a
        # symmetric cone averages the left/right detour modes into a
        # through-the-cone trajectory.
        _sk.vector_rotation_angle2_low  = 10.0
        _sk.vector_rotation_angle2_high = 10.0
        print("[env] fixed scenario: init error 120-140°, KOZ half-angle 20°, "
              "cone offset 10° (unimodal detour)")

    # Reset seed
    reset_seed(hparams.seed)

    # Fix data/env seed for lqr10
    if hparams.env == "lqr10" and hparams.rand_aggregate_seed is not None:
        np.random.seed(hparams.rand_aggregate_seed)
        random.seed(hparams.rand_aggregate_seed)

    if hparams.resume:
        # Restore model and agent
        mnet, hnet, agent, checkpoint, collector = reload_model(hparams)

        # Restore Logger
        logger = MonitorHnet(hparams, agent, mnet, hnet, collector)
        logger.load_stats(checkpoint)

        # Get num tasks we previously trained
        num_tasks_seen = checkpoint['num_tasks_seen']

    else:
        # Collect some random data
        collector = DataCollector(hparams)

        # Reuse the fixed normaliser of a previous run (norms.pt saved next to
        # its TensorBoard logs): every task in this run is then normalised
        # with exactly the same affine transform as that run.
        norms_path = getattr(hparams, "norms_path", None)
        if norms_path:
            payload = torch.load(norms_path, weights_only=True)
            collector.load_frozen_norms(payload["norms"], payload.get("diff_norms"))
            print(f"[norms] loaded frozen normalisation stats from {norms_path}")

        # Build model
        mnet, hnet = build_model(hparams)

        # RL Agent (MPC expert — also serves as data generator)
        agent = SafeAgent(hparams, mnet, collector=collector, hnet=hnet)

        # Lightweight NN policy that imitates the MPC expert
        policy = PolicyNet(
            state_dim=hparams.state_dim,
            action_dim=hparams.control_dim,
        ).to(hparams.device)
        policy_trainer = PolicyTrainer(policy, hparams)
        policy_trainer._dagger_n_iter = getattr(hparams, "dagger_n_iter", 5)
        nn_agent = NNPolicyAgent(hparams, policy, collector=collector)

        # Monitor
        logger = MonitorHnet(hparams, agent, mnet, hnet, collector)

        # Start from scratch
        num_tasks_seen = 0

    # Convert to cuda
    mnet.to(hparams.device)
    hnet.to(hparams.device)

    mparams = list(mnet.parameters())
    if len(mparams) > 0:
        print("MNET DEVICE:", mparams[0].device)

    hparams_ = list(hnet.parameters()) if hnet is not None else []
    if len(hparams_) > 0:
        print("HNET DEVICE:", hparams_[0].device)

    # Random Policy
    rand_pi = RandomAgent(hparams)

    # Start learning in environment
    envs = CLEnvHandler(hparams.env, hparams.seed)
    if hparams.resume:
        for tid in range(num_tasks_seen):
            envs.add_task(tid)

    for task_id in range(num_tasks_seen, hparams.num_tasks):
        # New Task with different friction
        env = envs.add_task(task_id, render=getattr(hparams, 'render', False))
        if hasattr(env, 'get_safety_filter'):
            agent.set_safety_filter(env.get_safety_filter())

        # Reset per-task DAGGER state: kappa/lambda curriculum restarts and the
        # rollout buffer is cleared so each task gets a fresh DAGGER pass.
        if not hparams.resume:
            policy_trainer.reset_per_task()

        print(f"Collecting some random data first for task {task_id}")
        x_t, _ = env.reset()
        rand_koz_hits = 0
        for it in range(hparams.init_rand_steps):
            # Clip to the env's actuator box: RandomAgent samples unbounded
            # randn, and unclipped commands both execute physically impossible
            # torques in the env (up to ~4x the box on spaceEnv) and store
            # out-of-range actions in the collector.  No-op for unbounded
            # action spaces (e.g. lqr).
            u = np.clip(rand_pi.act(x_t),
                        env.action_space.low.reshape(-1, 1),
                        env.action_space.high.reshape(-1, 1))
            x_tt, _, terminated, truncated, info = env.step(u.reshape(env.action_space.shape))
            done = terminated or truncated
            if info.get("keep_out_violation"):
                rand_koz_hits += 1
            collector.add(x_t, u, x_tt, task_id)
            x_t = x_tt
            logger.data_aggregate_step(x_tt, task_id, it)
            logger._log_safety(info, task_id, global_step=it)
            if done:
                x_t, _ = env.reset()
        print(f"  [random phase] task {task_id} koz_violations={rand_koz_hits}/{hparams.init_rand_steps}")
        if logger.writer is not None:
            logger.writer.add_scalar(f"random_phase/task_{task_id}/koz_violations", rand_koz_hits, task_id)

        # Freeze the normalisation statistics on the first task's random data
        # (actions get the identity transform — the box is already [-1,1]³).
        # From here on the normalised coordinate system is fixed for the whole
        # run: normalised buffers/closures can never go stale and the hnet sees
        # one stationary input distribution across tasks.  For later tasks the
        # call only assigns the already-frozen stats.  Saved to the run dir so
        # other runs can reuse the exact same transform via hparams.norms_path.
        if getattr(hparams, "freeze_norms", False) and hparams.normalize_xu:
            was_frozen = collector.frozen
            collector.freeze_norms(task_id)
            if not was_frozen:
                norms_file = os.path.join(logger.tflog_dir, "norms.pt")
                torch.save({"norms": collector._frozen_norms,
                            "diff_norms": collector._frozen_diff_norms}, norms_file)
                print(f"[norms] frozen normalisation stats (task {task_id}) -> {norms_file}")

        # Augment Model, instantiate optimizers/regularizer targets
        trainer_misc = augment_model(task_id, mnet, hnet, collector, hparams)

        # Pre-training baseline eval fires once per task, right before the
        # first policy training (needs the per-task norms, which exist only
        # after the first dynamics update inside the loop below).
        baseline_eval_done = False

        # Interact with the environment
        x_t, _ = env.reset()
        agent.reset()
        for it in range(hparams.max_iteration):
            if it % hparams.dynamics_update_every == 0:
                # Train Dynamics Model
                ts = time.time()
                train_set, _ = collector.get_dataset(task_id)
                train(task_id, mnet, hnet, trainer_misc, logger, train_set, hparams)
                print("Training time", time.time() - ts)

                # Rebuild CBF/CLF fns now that per-task norms are finalised.
                if getattr(hparams, "policy_use_safety_loss", False):
                    if hparams.env == "half_cheetah_safe":
                        x_mu, x_std, a_mu, a_std = collector.norm(task_id)
                        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
                        policy_trainer.cbf_fn = make_cheetah_cbf_fn(
                            unwrapped.keep_out_zones, x_mu, x_std, a_mu, a_std)
                    elif hparams.env.startswith("spaceEnv"):
                        x_mu, x_std, a_mu, a_std = collector.norm(task_id)
                        _raw_env = env.unwrapped if hasattr(env, 'unwrapped') else env
                        policy_trainer.cbf_fn = make_space_cbf_fn(
                            x_mu, x_std, a_mu, a_std, _raw_env.inertia)
                        policy_trainer.clf_fn = make_space_clf_fn(
                            x_mu, x_std, a_mu, a_std, _raw_env.inertia)
                        # θ-margin extractor for safety-prioritised sampling
                        policy_trainer.margin_fn = make_space_margin_fn(x_mu, x_std)
                        # Synthetic KOZ-corridor states for the CBF penalty —
                        # the training data is all-safe, so without these the
                        # CBF loss never fires (baseline_33: loss_cbf == 0).
                        policy_trainer.boundary_sampler = \
                            make_space_boundary_sampler(x_mu, x_std)
                        # Mask control-infeasible corridor states out of that
                        # penalty (baseline_34: viol frac pinned at ~0.6).
                        policy_trainer.cbf_feasible_fn = make_space_cbf_feasible_fn(
                            x_mu, x_std, _raw_env.inertia,
                            eps=getattr(hparams, "policy_cbf_eps_train", 0.0))

                # Train NN policy — wait until MPC has collected meaningful data.
                # Use the combined dataset (BC base + DAGGER expert buffer) so
                # the policy always trains on correct expert labels.
                if it >= getattr(hparams, "policy_train_start", 0):
                    # Round-0 baseline: evaluate the UNTRAINED policy (for
                    # task 0; for later tasks: the pre-task policy) with and
                    # without the filter, under the same dagger_eval_* tags.
                    # The validation charts then start from the untrained
                    # level, making the decline of filter activations and
                    # unfiltered KOZ violations across DAGGER rounds explicit.
                    if not baseline_eval_done:
                        baseline_eval_done = True
                        bl_env = logger.eval_envs.get_env(task_id)
                        nn_agent.cache_state_norm(task_id)
                        if hasattr(bl_env, 'get_safety_filter'):
                            nn_agent.set_safety_filter(bl_env.get_safety_filter())
                        print(f"  [baseline eval] task {task_id} — untrained policy (round 0)")
                        _bl_f = _eval_nn_policy(
                            nn_agent, bl_env, task_id, logger.writer,
                            policy_trainer._step,
                            n_episodes=getattr(hparams, "dagger_val_eps_filtered", 10),
                            tag_prefix="dagger_eval_filtered",
                            disable_filter=False,
                        )
                        _bl_u = _eval_nn_policy(
                            nn_agent, bl_env, task_id, logger.writer,
                            policy_trainer._step,
                            n_episodes=getattr(hparams, "dagger_val_eps_unfiltered", 40),
                            tag_prefix="dagger_eval_unfiltered",
                            disable_filter=True,
                        )
                        _log_filter_success(logger.writer, task_id,
                                            policy_trainer._step, _bl_f, _bl_u)
                    # BC base = MPC-phase rows only.  The random phase's
                    # "labels" are RandomAgent noise (unclipped randn — 27% of
                    # the base in baseline_27 with u_target_norm_max pinned at
                    # 3.98, i.e. 4x the actuator box), and the magnitude-
                    # weighted imitation loss up-weights exactly those rows.
                    # The dynamics model keeps the full train_set: wide
                    # excitation helps it.  Base is all-expert by construction
                    # (DAGGER rollouts never write to the collector); the
                    # trainer appends its expert-labelled buffer on top.
                    bc_base, _ = collector.get_dataset(
                        task_id, skip_first_n=hparams.init_rand_steps)
                    policy_train_set, sample_w = policy_trainer._make_policy_train_set(
                        bc_base, collector, task_id)
                    policy_trainer.train(policy_train_set, writer=logger.writer,
                                         sample_weights=sample_w)
                    if logger.writer is not None:
                        n_dyn = bc_base.tensors[0].shape[0]
                        n_dag = len(policy_trainer._dag_states)
                        logger.writer.add_scalar("policy/train_set_n_dyn", n_dyn, logger.env_iter)
                        logger.writer.add_scalar("policy/train_set_n_dag", n_dag, logger.env_iter)
            if getattr(hparams, 'render', False):
                env.render()
            # Cache the mainnet weight
            agent.cache_hnet(task_id)
            # Run MPC
            u_t = agent.act(x_t, task_id=task_id).detach().cpu().numpy()
            x_tt, reward, terminated, truncated, info = env.step(u_t.reshape(env.action_space.shape))
            done = terminated or truncated
                
            # Update the dataset of the env in which we're training 
            collector.add(x_t, u_t, x_tt, task_id)
            x_t = x_tt

            if done:
                x_t, _ = env.reset()
                agent.reset()
  
            logger.env_step(x_tt, reward, done, info, task_id)

            # Periodically run one eval episode with the NN policy and log reward.
            # Uses a dedicated env (logger.eval_envs) instead of the live training
            # `env` — reset()/step() here must never disturb the training rollout's
            # in-progress episode or desync it from the outer loop's `x_t`.
            if it > 0 and it % hparams.eval_env_run_every == 0:
                eval_env = logger.eval_envs.get_env(task_id)
                nn_agent.cache_state_norm(task_id)
                if hasattr(eval_env, 'get_safety_filter'):
                    nn_agent.set_safety_filter(eval_env.get_safety_filter())
                _eval_nn_policy(
                    nn_agent, eval_env, task_id, logger.writer, logger.env_iter,
                    n_episodes=getattr(hparams, "run_eval_env_eps", 1),
                )

            # DAGGER refinement (Algorithm 1) — run after the initial BC phase.
            dagger_every = getattr(hparams, "dagger_every", 0)
            if (dagger_every > 0
                    and it >= getattr(hparams, "policy_train_start", 0)
                    and it > 0
                    and it % dagger_every == 0
                    and policy_trainer._dagger_iter < getattr(hparams, "dagger_n_iter", 5)):
                nn_agent.cache_state_norm(task_id)

                # Capture norms for the closure — populated by cache_state_norm above.
                _x_mu  = nn_agent.x_mu   # (1, proc_dim) on device, or None
                _x_std = nn_agent.x_std
                _a_mu  = nn_agent.a_mu.cpu().numpy().flatten() if nn_agent.a_mu is not None else None
                _a_std = nn_agent.a_std.cpu().numpy().flatten() if nn_agent.a_std is not None else None

                def _preprocess_for_dagger(raw_obs):
                    import torch as _torch
                    x = _torch.tensor(raw_obs, dtype=_torch.float32,
                                      device=hparams.device).unsqueeze(0)
                    x = _preprocess_state_torch(x, hparams.env)
                    if nn_agent.normalize_xu and _x_mu is not None:
                        x = (x - _x_mu) / _x_std
                    return x

                # Rolls out in a dedicated env (logger.eval_envs), not the live
                # training `env` — see comment above the eval-env block.
                policy_trainer.dagger_update(
                    env=logger.eval_envs.get_env(task_id),
                    mpc_agent=agent,
                    collector=collector,
                    task_id=task_id,
                    preprocess_fn=_preprocess_for_dagger,
                    n_rollout=getattr(hparams, "dagger_n_rollout", 5),
                    max_ep_steps=1000,
                    writer=logger.writer,
                    a_mu=_a_mu,
                    a_std=_a_std,
                    filter_rollouts=getattr(hparams, "dagger_filter_rollouts", False),
                    bc_skip_first_n=hparams.init_rand_steps,
                    student_frac=getattr(hparams, "dagger_student_frac", 0.0),
                )

                # Post-DAGGER validation: run the *current* NN policy with and
                # without the safety filter, logged at the same policy step
                # (aligns on the x-axis with dagger/iter, dagger/kappa, ...).
                # The unfiltered pass answers "did DAGGER make the learned
                # policy itself safer", independent of what the QP filter
                # covers up — that's the whole point of tracking it separately
                # from policy_eval/* (which always runs filtered).
                val_env = logger.eval_envs.get_env(task_id)
                nn_agent.cache_state_norm(task_id)
                if hasattr(val_env, 'get_safety_filter'):
                    nn_agent.set_safety_filter(val_env.get_safety_filter())
                # Episode counts differ: unfiltered episodes are nearly free
                # and carry the high-variance safety signal; filtered ones
                # solve the QP every step (~6 s/episode) — see default_arg.
                _dv_f = _eval_nn_policy(
                    nn_agent, val_env, task_id, logger.writer, policy_trainer._step,
                    n_episodes=getattr(hparams, "dagger_val_eps_filtered", 10),
                    tag_prefix="dagger_eval_filtered",
                    disable_filter=False,
                )
                _dv_u = _eval_nn_policy(
                    nn_agent, val_env, task_id, logger.writer, policy_trainer._step,
                    n_episodes=getattr(hparams, "dagger_val_eps_unfiltered", 20),
                    tag_prefix="dagger_eval_unfiltered",
                    disable_filter=True,
                )
                _log_filter_success(logger.writer, task_id,
                                    policy_trainer._step, _dv_f, _dv_u)

                # Persist the distilled policy after every DAGGER round — the
                # mnet/hnet checkpoints never included it, so a crash (or the
                # end of the run) would otherwise lose the actual deliverable.
                torch.save(policy_trainer.policy.state_dict(),
                           os.path.join(logger.model_dir, "policy_last.pt"))

        augment_model_after(task_id, mnet, hnet, hparams, collector)

        # End-of-task policy snapshot (aligned with model_{task}.pt naming)
        if not hparams.resume:
            torch.save(policy_trainer.policy.state_dict(),
                       os.path.join(logger.model_dir, f"policy_{task_id}.pt"))

        # Save Model
        logger.save(task_id)

    envs.close()
    logger.writer.close()

def chunked_hnet(env, seed=None, savepath=None, play=False, render=False, run_name=None):
    # Hyperparameters
    hparams = HP(env, seed, savepath, run_name=run_name)
    hparams.model = "chunked_hnet"
    hparams.render = render

    hparams = Hparams.add_chunked_hnet_hparams(hparams)

    if play:
        play_model(hparams)
    else:
        run(hparams)


def hnet(env, seed=None, savepath=None, play=False, render=False, device="cpu",
         run_name=None, num_tasks=None, norms_path=None, fast_dagger=False,
         fixed_scenario=False):
    # Hyperparameters
    hparams = HP(env, seed, savepath, run_name=run_name)
    hparams.model = "hnet"
    hparams.render = render
    hparams.device = device
    if num_tasks is not None:
        hparams.num_tasks = num_tasks
    if norms_path is not None:
        hparams.norms_path = norms_path
    hparams.space_fixed_scenario = fixed_scenario

    if fast_dagger:
        # Shortened single-task profile to answer "does DAGGER internalise
        # safety?" (~1.5 h on cuda vs ~4-5 h/task for the full profile).
        # The pre-DAGGER phases are untouched: the expert must be competent
        # before its labels are worth imitating (policy_train_start=10000),
        # and the random phase feeds the dynamics model.  Savings come from
        # one task, a shorter DAGGER window, leaner rollouts and fewer
        # QP-filtered validation episodes.
        hparams.fast_dagger = True
        if num_tasks is None:
            hparams.num_tasks = 1
        hparams.max_iteration = 15000
        # Rounds span the whole trainable stretch (policy_train_start → end):
        hparams.dagger_n_iter = (hparams.max_iteration
                                 - hparams.policy_train_start) // hparams.dagger_every
        hparams.dagger_n_rollout = 3          # 3 rollouts/iter (0.4 → 1-2 student)
        hparams.dagger_val_eps_filtered = 10  # filter_fraction is the headline
                                              # decline metric — 5 eps was too
                                              # noisy for a rare-event rate
                                              # (unfiltered stays cheap at 40)

    hparams = Hparams.add_hnet_hparams(hparams)

    if play:
        play_model(hparams)
    else:
        run(hparams)

def hnet_si(env, seed=None, savepath=None, play=False, run_name=None):
    # Hyperparameters
    hparams = HP(env, seed, savepath, run_name=run_name)
    hparams.model = "hnet_si"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0.05
    hparams.grad_max_norm = 5

    if play:
        play_model(hparams)
    else:
        run(hparams)

def hnet_ewc(env, seed=None, savepath=None, play=False, run_name=None):
    # Hyperparameters
    hparams = HP(env, seed, savepath, run_name=run_name)
    hparams.model = "hnet_ewc"

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0.05
    hparams.ewc_weight_importance = True
    hparams.n_fisher = -1

    if play:
        play_model(hparams)
    else:
        run(hparams)

def hnet_mt(env, seed=None, savepath=None, play=False, render=False, run_name=None):
    # Hyperparameters
    hparams = HP(env, seed, savepath, run_name=run_name)
    hparams.model = "hnet_mt"
    hparams.render = render

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0
    hparams.plastic_prev_tembs = True

    if play:
        play_model(hparams)
    else:
        run(hparams)

def hnet_replay(env, seed=None, savepath=None, play=False, render=False, run_name=None):
    # Hyperparameters
    hparams = HP(env, seed, savepath, run_name=run_name)
    hparams.model = "hnet_replay"
    hparams.render = render

    hparams = Hparams.add_hnet_hparams(hparams)
    hparams.beta = 0.05
    hparams.grad_max_norm = 5
    hparams.plastic_prev_tembs = True


    if play:
        play_model(hparams)
    else:
        run(hparams)

if __name__ == "__main__":
    import fire
    fire.Fire({
        'hnet': hnet,
        'chunked_hnet': chunked_hnet,
        'hnet_si': hnet_si,
        'hnet_ewc': hnet_ewc,
    })
