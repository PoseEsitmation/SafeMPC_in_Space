import torch
import numpy as np


class GTCost():
    """Ground-truth planning cost used by the MPC/CEM planner.

    One cost branch per environment family kept in this repo: half_cheetah
    (incl. half_cheetah_safe), cartpole (+ cartpole_bin) and spaceEnv (+ _moi).
    """

    def __init__(self, clenv_name, state_dim, control_dim, reward_discount, device):
        self.env_name = clenv_name
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.reward_discount = reward_discount

        self.cartpole_x = [0, -5, 5]
        self.pole_length = [0.6, 0.8, 0.4, 1.0, 1.2, 0.7, 0.5, 0.9, 1.1, 1.3]

    def __call__(self, x, u, t, task_id):
        x = x.view(-1, self.state_dim)
        u = u.view(-1, self.control_dim)
        if self.env_name.startswith("half_cheetah"):
            reward_ctrl = -0.1 * (u ** 2).sum(dim=1)
            reward_run = x[:, 0]
            cost = - reward_run - reward_ctrl
        elif self.env_name == "cartpole_bin":
            l = 0.6
            a = torch.pow((x[:, 0] - self.cartpole_x[task_id] - l * torch.sin(x[:, 1])),
                          2) + torch.pow((-l * torch.cos(x[:, 1]) - l), 2)
            reward = torch.exp(-a/(l*l))
            reward -= 0.01 * torch.sum(torch.pow(u, 2), dim=-1)
            cost = -reward
        elif self.env_name == "cartpole":
            l = self.pole_length[task_id]
            a = torch.pow((x[:, 0] - l * torch.sin(x[:, 1])), 2) + \
                torch.pow((-l * torch.cos(x[:, 1]) - l), 2)
            reward = torch.exp(-a/(l*l))
            reward -= 0.01 * torch.sum(torch.pow(u, 2), dim=-1)
            cost = -reward
        elif self.env_name == "spaceEnv" or self.env_name.startswith("spaceEnv_"): #taken from the sat_env.py reward
            # x[:, 0] = qe_0 (error quaternion scalar, >=0 by convention)
            qe_0 = torch.clamp(x[:, 0], -1.0, 1.0)
            err_phi = 2.0 * torch.acos(qe_0)
            attitude_reward = torch.exp(-err_phi / (0.14 * 2.0 * np.pi))
            # torque penalty (scale_torque=2, torque_max=2*sqrt(3))
            torque_norm = torch.norm(u * 2.0, p=2, dim=-1) / (2.0 * np.sqrt(3))
            # KOZ penalty: x[:, 7] = theta_margin_norm, invert normalisation
            # theta_margin_norm = -1 + (theta_margin + pi/2) * 4 / (3*pi)
            theta_margin = (x[:, 7] + 1.0) * (3.0 * np.pi / 4.0) - np.pi / 2.0
            penalty_koz = torch.where(
                theta_margin <= 0,
                torch.full_like(theta_margin, 10.0),
                10.0 * torch.exp(-66.0 * theta_margin),
            )
            # progress penalty: -1 when attitude error increases (matches env)
            # x[:, 12] = qe_0_prev (set explicitly during planning rollouts)
            qe_0_prev = torch.clamp(x[:, 12], -1.0, 1.0)
            err_phi_prev = 2.0 * torch.acos(qe_0_prev)
            progress_penalty = (err_phi > err_phi_prev).float()
            # goal bonus: +9 when within 0.25 deg (matches env reward)
            goal_bonus = torch.where(
                err_phi <= 0.25 * np.pi / 180.0,
                torch.full_like(err_phi, 9.0),
                torch.zeros_like(err_phi),
            )
            reward = (attitude_reward - 0.05 * torque_norm - penalty_koz
                      - progress_penalty + goal_bonus)
            cost = -reward
        else:
            raise ValueError(f"GTCost: unsupported environment '{self.env_name}'")
        return cost

    def reward(self, x, u, t, task_id):
        cost = self.__call__(x, u, t, task_id)
        return -cost
