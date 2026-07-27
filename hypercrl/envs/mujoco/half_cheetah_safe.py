from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import cvxpy as cp
import mujoco
import numpy as np
import torch
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env

from hypercrl.control.safety_filter import CBF, CLF, SafetyFilter

logger = logging.getLogger(__name__)


class HalfCheetahKeepOutCBF(CBF):
    """Relative-degree-2 extended CBF for x-position keep-out zones.

    For each zone ``[x_min, x_max]`` the barrier is formed depending on
    which side the cheetah is currently on:

    Left approach (x_pos <= x_min):
        h_L  = x_min - x_pos       (positive = safe)
        H_L  = -x_vel + alpha * h_L
        Ḣ_L  = -ẍ_vel - alpha * x_vel  ≈  -(s @ u) - alpha * x_vel

    Right approach (x_pos >= x_max):
        h_R  = x_pos - x_max
        H_R  =  x_vel + alpha * h_R
        Ḣ_R  =  ẍ_vel + alpha * x_vel  ≈   (s @ u) + alpha * x_vel

    ``s`` is a uniform sensitivity vector approximating how each action
    dimension contributes to x-acceleration.  Tune ``x_accel_gain`` or
    replace ``H_dot_expr`` with a proper linearisation from the model.
    """

    def __init__(
        self,
        env: "HalfCheetahSafeEnv",
        alpha: float = 1.0,
        x_accel_gain: float = 0.5,
    ) -> None:
        self._env = env
        self.alpha = alpha
        self.x_accel_gain = x_accel_gain

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _x_pos(self) -> float:
        return float(self._env.data.qpos[0])

    def _pick_active_barrier(self, x_pos: float, x_vel: float):
        """Return (H_value, side) for the most constraining zone.

        ``side`` is ``'left'``, ``'right'``, or ``None`` (no zones).
        """
        zones = self._env.keep_out_zones
        if not zones:
            return float("inf"), None

        min_H = float("inf")
        active_side = None

        for x_min, x_max in zones:
            if x_pos <= x_min:
                H = -x_vel + self.alpha * (x_min - x_pos)
                side = "left"
            elif x_pos >= x_max:
                H = x_vel + self.alpha * (x_pos - x_max)
                side = "right"
            else:
                # Inside zone: choose the boundary we are closer to.
                if (x_pos - x_min) <= (x_max - x_pos):
                    H = -x_vel + self.alpha * (x_min - x_pos)  # negative
                    side = "left"
                else:
                    H = x_vel + self.alpha * (x_pos - x_max)   # negative
                    side = "right"

            if H < min_H:
                min_H = H
                active_side = side

        return min_H, active_side

    # ------------------------------------------------------------------
    # CBF interface
    # ------------------------------------------------------------------

    def _z_pos(self) -> float:
        return float(self._env.data.qpos[1])

    def H(self, state: np.ndarray) -> float:
        if self._z_pos() >= self._env.zone_height:
            return float("inf")  # cheetah has cleared the obstacle
        x_vel = float(state[0])
        h_val, _ = self._pick_active_barrier(self._x_pos(), x_vel)
        return h_val

    def H_dot_expr(self, state: np.ndarray, u_var: cp.Variable) -> cp.Expression:
        if self._z_pos() >= self._env.zone_height:
            return cp.Constant(1.0)  # above obstacle — no constraint

        x_vel = float(state[0])
        _, side = self._pick_active_barrier(self._x_pos(), x_vel)

        if side is None:
            # No keep-out zones — trivially satisfied.
            return cp.Constant(1.0)

        n_act = u_var.shape[0]
        s = np.full(n_act, self.x_accel_gain / n_act)

        if side == "left":
            # Ḣ_L ≈ -(s @ u) - alpha * x_vel
            return -(s @ u_var) - self.alpha * x_vel
        else:
            # Ḣ_R ≈  (s @ u) + alpha * x_vel
            return (s @ u_var) + self.alpha * x_vel


def make_cheetah_cbf_fn(
    zones: List[Tuple[float, float]],
    x_mu: torch.Tensor,
    x_std: torch.Tensor,
    a_mu: torch.Tensor,
    a_std: torch.Tensor,
    alpha: float = 1.0,
    x_accel_gain: float = 0.5,
) -> Callable:
    """Torch-differentiable batched version of HalfCheetahKeepOutCBF.H_dot_expr.

    Used by the CBF term of the imitation loss, so it takes the states the
    PolicyTrainer stores: preprocessed and collector-normalised.  Layout after
    preprocessing (19 dims):
        [0]  z_pos
        [1]  cos(root_angle)
        [2]  sin(root_angle)
        [3-8] joint angles
        [9]  qvel[0]  ← global x-velocity  (what the CBF needs)
        [10-17] remaining velocities
        [18] x_pos   ← global x-position   (what the CBF needs)

    Both x_vel and x_pos are further normalised by (x_mu, x_std); the action
    is normalised by (a_mu, a_std).  This factory captures those statistics so
    the returned fn can denormalise on the fly.

    H_dot (linear in u, as in HalfCheetahKeepOutCBF.H_dot_expr):
        left  approach: -(s @ u_phys) - alpha * x_vel
        right approach:  (s @ u_phys) + alpha * x_vel
    where s = x_accel_gain / n_actions (uniform sensitivity).
    """
    # Flatten: finalize() stores norms as (1, dim) after .T — index into dim axis.
    x_mu_f = x_mu.flatten()
    x_std_f = x_std.flatten()
    xvel_mu  = float(x_mu_f[9])
    xvel_std = float(x_std_f[9])
    xpos_mu  = float(x_mu_f[18])
    xpos_std = float(x_std_f[18])

    def cbf_fn(state_norm: torch.Tensor, action_norm: torch.Tensor) -> torch.Tensor:
        dev = state_norm.device

        # Denormalise the two quantities the CBF depends on.
        x_vel = state_norm[:, 9]  * xvel_std + xvel_mu   # global x-velocity
        x_pos = state_norm[:, 18] * xpos_std + xpos_mu   # global x-position

        # Denormalise action to physical space (a_mu/a_std are (1,n_act) — flatten to (n_act,)).
        u_phys = action_norm * a_std.flatten().to(dev) + a_mu.flatten().to(dev)

        n_act = u_phys.shape[1]
        s     = x_accel_gain / n_act
        s_u   = s * u_phys.sum(dim=1)          # s @ u  (uniform s)

        h_dot_left  = -s_u - alpha * x_vel
        h_dot_right =  s_u + alpha * x_vel

        min_h_dot = torch.full_like(x_vel, float("inf"))
        for x_min, x_max in zones:
            x_min_t = torch.tensor(x_min, dtype=x_vel.dtype, device=dev)
            x_max_t = torch.tensor(x_max, dtype=x_vel.dtype, device=dev)

            on_left  = (x_pos <= x_min_t).float()
            on_right = (x_pos >= x_max_t).float()
            in_zone  = 1.0 - on_left - on_right

            h_dot = (
                on_left  * h_dot_left
                + on_right * h_dot_right
                + in_zone  * torch.minimum(h_dot_left, h_dot_right)
            )
            min_h_dot = torch.minimum(min_h_dot, h_dot)

        return min_h_dot   # positive ⇒ constraint met; negative ⇒ violation

    return cbf_fn


class HalfCheetahSafeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    HalfCheetah with keep-out zones.
    keep_out_zones: list of (x_min, x_max) tuples — forbidden x-position regions.
    Entering a zone applies a large negative penalty and terminates the episode.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, keep_out_zones=None, penalty=100.0, forward_reward_weight=1.0,
                 zone_height=0.7, render_mode=None):

         # Store the forbidden zones — if none given, use empty list (no restrictions)
        self.keep_out_zones = keep_out_zones if keep_out_zones is not None else []
        self.penalty = penalty
        self.forward_reward_weight = forward_reward_weight
        self.zone_height = zone_height  # torso z below this = inside zone (must jump over)
        # Initialize x-position tracker to None — will be set on first step
        self.xposbefore = None

        # Define what the agent can observe — 18 numbers describing the cheetah's state
        # low=-inf, high=inf means no clipping on observation values
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64
        )

        # Initialize EzPickle with the same arguments — required for saving/loading the env
        utils.EzPickle.__init__(
            self, keep_out_zones=keep_out_zones,
            penalty=penalty, forward_reward_weight=forward_reward_weight,
            zone_height=zone_height, render_mode=render_mode,
        )

         # Initialize the MuJoCo base environment
        # 'half_cheetah.xml' is the file describing the robot's physical structure
        # 5 is the frame_skip — physics runs 5 sub-steps per agent step
        mujoco_env.MujocoEnv.__init__(
            self, 'half_cheetah.xml', 5,
            observation_space=observation_space,
            render_mode=render_mode
        )

    def _in_keep_out_zone(self, xpos, zpos):
        """Violation only when inside the x-range AND below zone_height (not jumped over)."""
        for (x_min, x_max) in self.keep_out_zones:
            if x_min <= xpos <= x_max and zpos < self.zone_height:
                return True
        return False

    def _is_flipped(self) -> bool:
        """True when the torso pitch exceeds 90 degrees — cheetah is upside-down."""
        return bool(np.abs(self.data.qpos[2]) > np.pi / 2)

    def step(self, action):
        self.xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        zposafter = self.data.qpos[1]  # torso height
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run  = self.forward_reward_weight * (xposafter - self.xposbefore) / self.dt

        min_dist = self._min_zone_dist(float(xposafter), float(zposafter))

        if self._in_keep_out_zone(xposafter, zposafter):
            reward     = -self.penalty
            terminated = True
            info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                keep_out_violation=True,
                flipped=False,
                violated_at=float(xposafter),
                min_zone_dist=0.0,
            )
        elif self._is_flipped():
            reward     = -self.penalty
            terminated = True
            info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                keep_out_violation=False,
                flipped=True,
                min_zone_dist=min_dist,
            )
        else:
            reward     = reward_ctrl + reward_run
            terminated = False
            info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                keep_out_violation=False,
                flipped=False,
                min_zone_dist=min_dist,
            )

        return ob, reward, terminated, False, info

    def _min_zone_dist(self, x_pos: float, z_pos: float) -> float:
        """Horizontal distance to nearest zone; infinite when cheetah is above zone_height."""
        if not self.keep_out_zones or z_pos >= self.zone_height:
            return float("inf")
        dists = []
        for x_min, x_max in self.keep_out_zones:
            if x_pos < x_min:
                dists.append(x_min - x_pos)
            elif x_pos > x_max:
                dists.append(x_pos - x_max)
            else:
                dists.append(0.0)
        return min(dists)

    def render(self):
        if self.render_mode == "human" and self.keep_out_zones:
            viewer = self.mujoco_renderer._viewers.get("human")
            if viewer is not None:
                self._patch_viewer_if_needed(viewer)
                self._add_zone_markers(viewer)
        return super().render()

    def _patch_viewer_if_needed(self, viewer) -> None:
        """Replace gymnasium's broken _add_marker_to_scene with one that uses mjv_initGeom.

        gymnasium's default implementation sets g.texid which was removed in newer mujoco.
        """
        if getattr(viewer, '_zone_marker_patched', False):
            return

        def _add_marker_to_scene(marker: dict):
            if viewer.scn.ngeom >= viewer.scn.maxgeom:
                return
            g = viewer.scn.geoms[viewer.scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                type=int(marker.get("type", mujoco.mjtGeom.mjGEOM_BOX)),
                size=np.asarray(marker.get("size", np.ones(3) * 0.1), dtype=np.float64).ravel(),
                pos=np.asarray(marker.get("pos", np.zeros(3)), dtype=np.float64).ravel(),
                mat=np.asarray(marker.get("mat", np.eye(3)), dtype=np.float64).ravel(),
                rgba=np.asarray(marker.get("rgba", np.ones(4)), dtype=np.float32).ravel(),
            )
            label = marker.get("label", "")
            if label:
                g.label = label
            viewer.scn.ngeom += 1

        viewer._add_marker_to_scene = _add_marker_to_scene
        viewer._zone_marker_patched = True

    def _add_zone_markers(self, viewer) -> None:
        """Queue ground-level red obstacles for every keep-out zone."""
        half_h = self.zone_height / 2.0
        for x_min, x_max in self.keep_out_zones:
            cx = (x_min + x_max) / 2.0
            half_width = (x_max - x_min) / 2.0
            viewer.add_marker(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                pos=np.array([cx, 0.0, half_h]),
                size=np.array([half_width, 0.05, half_h]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.1, 0.1, 0.5], dtype=np.float32),
                label=f"zone [{x_min:.1f},{x_max:.1f}]",
            )

    def _get_obs(self):
        return np.concatenate([
            (self.data.qpos.flat[:1] - self.xposbefore) / self.dt,  # [0]  x_vel
            self.data.qpos.flat[1:],                                  # [1-8] z_pos, angle, joints
            self.data.qvel.flat,                                      # [9-17] velocities
            self.data.qpos.flat[:1],                                  # [18]  x_pos (global)
        ])
    
    # Reset function — called at the start of every new episode
    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + \
            self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.xposbefore = np.copy(self.data.qpos[0])
        return self._get_obs()

    # ------------------------------------------------------------------
    # Safety filter interface
    # ------------------------------------------------------------------

    def get_cbf(self, alpha: float = 1.0, x_accel_gain: float = 0.5) -> Optional[HalfCheetahKeepOutCBF]:
        """Return a CBF for the current keep-out zones, or None if there are none."""
        if not self.keep_out_zones:
            return None
        return HalfCheetahKeepOutCBF(self, alpha=alpha, x_accel_gain=x_accel_gain)

    def get_clf(self) -> None:
        """CLF not yet defined for HalfCheetah; returns None (filter runs CBF-only)."""
        return None

    def get_safety_filter(
        self,
        cbf_epsilon: float = 0.0,
        clf_rho: float = 1e3,
        alpha: float = 1.0,
        x_accel_gain: float = 0.5,
    ) -> Optional[SafetyFilter]:
        """Convenience constructor: returns a ready-to-use SafetyFilter or None."""
        cbf = self.get_cbf(alpha=alpha, x_accel_gain=x_accel_gain)
        if cbf is None:
            return None
        return SafetyFilter(
            cbf=cbf,
            clf=self.get_clf(),
            u_max=1.0,
            control_dim=self.action_space.shape[0],
            cbf_epsilon=cbf_epsilon,
            clf_rho=clf_rho,
        )