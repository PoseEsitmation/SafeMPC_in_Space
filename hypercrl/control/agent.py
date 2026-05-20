from __future__ import annotations

import numpy as np
import torch
from typing import Optional, Protocol, Union, runtime_checkable

from .cem import CEM
from .mppi import MPPI, PDDM
from .lqr import LQR
from .grad import GradPlan
from .manual import Manual
from .reward import GTCost


@runtime_checkable
class SafetyFilterProtocol(Protocol):
    def filter(
        self,
        state: np.ndarray,
        u_proposed: np.ndarray,
    ) -> np.ndarray: ...


def quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    assert q0.shape == q1.shape
    assert q0.shape[-1] == 4
    assert q1.shape[-1] == 4

    w0 = q0[..., 0]
    x0 = q0[..., 1]
    y0 = q0[..., 2]
    z0 = q0[..., 3]

    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 + y0 * w1 + z0 * x1 - x0 * z1
    z = w0 * z1 + z0 * w1 + x0 * y1 - y0 * x1
    q = torch.stack([w, x, y, z], dim=-1)
    q = q / q.norm(2, dim=-1, keepdim=True)
    assert q.shape == q0.shape
    return q


class Agent:
    def __init__(self, hparams) -> None:
        self.model_name: str = hparams.model
        self.env_name: str = hparams.env
        self.control_dim: int = hparams.control_dim
        self.state_dim: int = hparams.state_dim
        self.dnn_out: str = hparams.dnn_out
        self.reward_discount: float = hparams.reward_discount
        self._cost: GTCost = GTCost(
            self.env_name, self.state_dim, self.control_dim,
            self.reward_discount, hparams.device,
        )

    def act(self, state: Union[torch.Tensor, np.ndarray],
            task_id: Optional[int] = None) -> Union[torch.Tensor, np.ndarray]:
        pass


class RandomAgent(Agent):
    def __init__(self, hparams) -> None:
        super().__init__(hparams)

    def act(self, state: Union[torch.Tensor, np.ndarray],
            task_id: Optional[int] = None) -> np.ndarray:
        return np.random.randn(self.control_dim, 1)


class MPC(Agent):
    def __init__(self, hparams, model, envs=None, collector=None,
                 likelihood=None, hnet=None) -> None:
        super().__init__(hparams)
        self.model = model
        self.hnet = hnet
        self.envs = envs
        self.collector = collector
        self.device = hparams.device
        self.out_var: bool = hparams.out_var
        self.normalize_xu: bool = hparams.normalize_xu if collector is not None else False
        self.gt_dynamic: bool = hparams.gt_dynamic
        self.control_type: str = hparams.control

        if self.model_name.startswith("hnet") or self.model_name == "chunked_hnet":
            self.reset_hnet()

        if hparams.control == "mpc-cem":
            self.control = CEM(
                self._dynamics, self._cost, hparams.state_dim, hparams.control_dim,
                num_samples=hparams.n_sim_particles,
                num_elite=hparams.num_cem_elites,
                num_iterations=hparams.n_sim_steps,
                horizon=hparams.horizon,
                device=hparams.device,
                u_min=None,
                u_max=None,
                choose_best=True,
                init_cov_diag=hparams.mag_noise,
            )
        elif hparams.control == "mpc-mppi":
            noise_sigma = (
                torch.eye(hparams.control_dim,
                          device=hparams.device, dtype=torch.float32)
                * hparams.mag_noise
            )
            self.control = MPPI(
                self._dynamics, self._cost, hparams.state_dim, noise_sigma,
                num_samples=hparams.n_sim_particles,
                num_iter=hparams.n_sim_steps,
                horizon=hparams.horizon,
                lambda_=1 / hparams.pddm_kappa,
                device=hparams.device,
                u_min=None,
                u_max=None,
            )
        elif hparams.control == "mpc-pddm":
            self.control = PDDM(
                self._dynamics, self._cost, hparams.state_dim, hparams.control_dim,
                hparams.horizon, hparams.n_sim_particles, hparams.pddm_beta,
                hparams.pddm_kappa, hparams.mag_noise, hparams.device,
            )
        elif hparams.control == "mpc-grad":
            self.control = GradPlan(
                self._dynamics, self._cost, hparams.state_dim, hparams.control_dim,
                hparams.n_sim_particles, hparams.n_sim_steps, hparams.horizon, hparams.device,
            )
        elif hparams.control == "mpc-lqr":
            self.control = LQR(hparams.state_dim,
                               hparams.control_dim, hparams.horizon)
        elif hparams.control == "manual":
            self.control = Manual(
                hparams.env, hparams.state_dim, hparams.control_dim,
                hparams.horizon, self._dynamics, hparams.device,
            )

    def cache_hnet(self, task_id: int) -> None:
        weights = self.hnet(task_id)
        self._cached_weights = [w.detach() for w in weights]

    def reset_hnet(self) -> None:
        self._cached_weights = None

    def cache_state_norm(self, task_id: int) -> None:
        if self.normalize_xu:
            x_mu, x_std, a_mu, a_std = self.collector.norm(task_id)
            self.x_mu, self.x_std = x_mu.to(self.device), x_std.to(self.device)
            self.a_mu, self.a_std = a_mu.to(self.device), a_std.to(self.device)

    def _dynamics(self, x: torch.Tensor, u: torch.Tensor,
                  task_id: Optional[int]) -> torch.Tensor:
        x = x.view(-1, self.state_dim)
        u = u.view(-1, self.control_dim)
        xcopy = x.clone()

        # State preprocessing
        if self.env_name.startswith("inverted_pendulum") or self.env_name.startswith("cartpole"):
            x = torch.cat((x[:, 0:1], torch.cos(x[:, 1:2]),
                          torch.sin(x[:, 1:2]), x[:, 2:]), dim=-1)
        elif self.env_name in ["half_cheetah_body", "hopper"]:
            x = torch.cat((x[:, 1:2], torch.cos(x[:, 2:3]),
                          torch.sin(x[:, 2:3]), x[:, 3:]), dim=-1)
        elif self.env_name == "door":
            x = torch.cat((x[:, 0:-1], torch.cos(x[:, -1:]),
                          torch.sin(x[:, -1:])), dim=-1)
        elif self.env_name == "door_pose":
            x = torch.cat((
                x[:, 0:-2],
                torch.cos(x[:, -2:-1]), torch.sin(x[:, -2:-1]),
                torch.cos(x[:, -1:]), torch.sin(x[:, -1:]),
            ), dim=-1)

        # FIXME: REMOVE THIS (now DEBUG ONLY)
        if self.gt_dynamic:
            if self.env_name == "pendulum":
                th = torch.atan2(x[:, 1], x[:, 0]).view(-1, 1)
                thdot = x[:, 2].view(-1, 1)

                g = 10
                m = 1
                l = 1
                dt = 0.05
                u = torch.clamp(u, -2, 2)

                newthdot = thdot + \
                    (-3 * g / (2 * l) * torch.sin(th + np.pi) +
                     3.0 / (m * l ** 2) * u) * dt
                newth = th + newthdot * dt
                newthdot = torch.clamp(newthdot, -8, 8)

                xx_gt = torch.cat(
                    (torch.cos(newth), torch.sin(newth), newthdot), dim=1)

        if self.normalize_xu:
            x = (x - self.x_mu) / self.x_std
            u = (u - self.a_mu) / self.a_std

        if self.model_name in ["single", "finetune", "coreset", "pnn", "ewc", "si", "multitask"]:
            xx = self.model(x, u, task_id)
        elif self.model_name.startswith("hnet") or self.model_name == "chunked_hnet":
            weights = self.hnet(
                task_id) if self._cached_weights is None else self._cached_weights
            xu = torch.cat((x, u), dim=-1)
            xx = self.model.forward(xu, weights)

        # For probabilistic output, select the mean
        if self.out_var:
            xx, _ = torch.split(xx, xx.size(-1) // 2, dim=-1)

        # (deprecated) Un-normalize output
        if self.dnn_out != "diff" and self.normalize_xu:
            xx = xx * self.x_std + self.x_mu

        # Compensate diff
        if self.env_name in ["half_cheetah_body", "hopper"] and self.dnn_out == "diff":
            xx = torch.cat((xx[:, 0:1], xcopy[:, 1:] + xx[:, 1:]), dim=-1)
        elif self.env_name == "door_pose" and self.dnn_out == "diff":
            xx = torch.cat((
                xcopy[:, 0:3] + xx[:, 0:3],
                quat_mul(xcopy[:, 3:7], xx[:, 3:7]),
                xcopy[:, 7:] + xx[:, 7:],
            ), dim=-1)
        elif self.dnn_out == "diff":
            xx = xcopy + xx

        if self.gt_dynamic:
            print((xx_gt - xx).mean(dim=0))
            return xx_gt
        return xx

    def reset(self) -> None:
        self.control.reset()

    def act(self, state: Union[torch.Tensor, np.ndarray],
            task_id: Optional[int] = None,
            first_action: bool = True) -> torch.Tensor:
        self.model.eval()
        if self.control_type != "manual":
            self.cache_state_norm(task_id)
        with torch.no_grad():
            cmd: torch.Tensor = self.control.command(
                state, task_id, first_action)
        return cmd


class SafeAgent(Agent):
    """MPC agent with an optional safety filter between act() and env.step().

    When ``safety_filter`` is None the agent is behaviourally identical to MPC.
    The filter is env-owned and passed in here; swap it per task via
    ``set_safety_filter(env.get_cbf(), env.get_clf())``.
    """

    def __init__(
        self,
        hparams,
        model,
        safety_filter: Optional[SafetyFilterProtocol] = None,
        envs=None,
        collector=None,
        likelihood=None,
        hnet=None,
    ) -> None:
        super().__init__(hparams)
        self.mpc: MPC = MPC(
            hparams, model,
            envs=envs, collector=collector, likelihood=likelihood, hnet=hnet,
        )
        self.safety_filter: Optional[SafetyFilterProtocol] = safety_filter
        print(
            f"Initialized SafeAgent with safety filter: {self.safety_filter}")

    # --- delegate MPC internals accessed directly by tools/loggers ---

    @property
    def _dynamics(self):
        return self.mpc._dynamics

    @property
    def model(self):
        return self.mpc.model

    @property
    def hnet(self):
        return self.mpc.hnet

    # --- delegate hnet / normalisation helpers to the inner MPC ---

    def cache_hnet(self, task_id: int) -> None:
        self.mpc.cache_hnet(task_id)

    def reset_hnet(self) -> None:
        self.mpc.reset_hnet()

    def cache_state_norm(self, task_id: int) -> None:
        self.mpc.cache_state_norm(task_id)

    def reset(self) -> None:
        self.mpc.reset()

    def set_safety_filter(self, safety_filter: Optional[SafetyFilterProtocol]) -> None:
        self.safety_filter = safety_filter

    def act(
        self,
        state: Union[torch.Tensor, np.ndarray],
        task_id: Optional[int] = None,
        first_action: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        u_proposed: torch.Tensor = self.mpc.act(
            state, task_id=task_id, first_action=first_action)

        if self.safety_filter is None:
            return u_proposed

        state_np: np.ndarray = (
            state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else state
        )
        u_np: np.ndarray = u_proposed.detach().cpu().numpy()
        return self.safety_filter.filter(state_np, u_np)


class RollOut:
    def __init__(self, hparams, model, collector) -> None:
        self.model = model
        self.collector = collector

        self.n_samples: int = hparams.n_sim_particles
        self.device = hparams.device

        self.x_dim: int = hparams.state_dim
        self.a_dim: int = hparams.control_dim
        self.horizon: int = hparams.horizon
        self.propagation = hparams.propagation
        self.dnn_out: str = hparams.dnn_out

    def predict(self, x_t: np.ndarray, actions: np.ndarray, task_id: int) -> None:
        raise NotImplementedError

    def plot_rollout(self, env, x_t: np.ndarray, actions: np.ndarray, task_id: int) -> None:
        raise NotImplementedError
