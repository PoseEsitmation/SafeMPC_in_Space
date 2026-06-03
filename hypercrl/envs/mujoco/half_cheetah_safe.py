from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env

from hypercrl.control.safety_filter import CBF, CLF, SafetyFilter


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

    def H(self, state: np.ndarray) -> float:
        x_vel = float(state[0])
        h_val, _ = self._pick_active_barrier(self._x_pos(), x_vel)
        return h_val

    def H_dot_expr(self, state: np.ndarray, u_var: cp.Variable) -> cp.Expression:
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

    def __init__(self, keep_out_zones=None, penalty=100.0, render_mode=None):
        
         # Store the forbidden zones — if none given, use empty list (no restrictions)
        self.keep_out_zones = keep_out_zones if keep_out_zones is not None else []
        self.penalty = penalty
        # Initialize x-position tracker to None — will be set on first step
        self.xposbefore = None

        # Define what the agent can observe — 18 numbers describing the cheetah's state
        # low=-inf, high=inf means no clipping on observation values
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )

        # Initialize EzPickle with the same arguments — required for saving/loading the env
        utils.EzPickle.__init__(
            self, keep_out_zones=keep_out_zones,
            penalty=penalty, render_mode=render_mode
        )

         # Initialize the MuJoCo base environment
        # 'half_cheetah.xml' is the file describing the robot's physical structure
        # 5 is the frame_skip — physics runs 5 sub-steps per agent step
        mujoco_env.MujocoEnv.__init__(
            self, 'half_cheetah.xml', 5,
            observation_space=observation_space,
            render_mode=render_mode
        )

    def _in_keep_out_zone(self, xpos):
        for (x_min, x_max) in self.keep_out_zones:
            # Check if xpos falls within this zone's boundaries
            if x_min <= xpos <= x_max:
                return True
        return False

    def step(self, action):
          # Record the cheetah's x-position BEFORE the action is applied
        self.xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0] # Record the cheetah's x-position AFTER the action is applied
        ob = self._get_obs()

        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run  = (xposafter - self.xposbefore) / self.dt

        if self._in_keep_out_zone(xposafter):
            print(f"[KeepOut] violation at x={xposafter:.3f}  zones={self.keep_out_zones}")
            reward     = -self.penalty
            terminated = True
            info = dict(
                 reward_run=reward_run,        # what running reward would have been
                reward_ctrl=reward_ctrl,      # what control penalty would have been
                keep_out_violation=True,      # flag that a violation occurred
                violated_at=float(xposafter) # exact x-position where violation happened
            )
        else:
            reward     = reward_ctrl + reward_run
            terminated = False
            info = dict(
                reward_run=reward_run,
                reward_ctrl=reward_ctrl,
                keep_out_violation=False
            )

        return ob, reward, terminated, False, info

    def _get_obs(self):
          # Combine three sources of information into one flat array:
        return np.concatenate([
            (self.data.qpos.flat[:1] - self.xposbefore) / self.dt,
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
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