"""
Satellite attitude control environment — Phase 1 (with Keep-Out Zone).

State vector (13,):
    [qe(4), omega_e(3), theta_margin(1), theta(1), relative_n_avoid_inb(3), qe_0_prev(1)]

Action space (3,):
    Normalised torques [-1, 1] along body axes, scaled by scale_torque [Nm].

Dependencies:
    subfunctions_att_constraints.py  (must be in the same directory or on PYTHONPATH)
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

from .subfunctions_att_constraints import KeepOutZone
from .subfunctions_att_constraints import generate_avoid_vector_in_i_for_1Fzone_phase1_v2

# ---------------------------------------------------------------------------
# Physical / simulation constants
# ---------------------------------------------------------------------------

deg2rad = np.pi / 180
rad2deg = 180 / np.pi

scale_torque = 2        # [Nm]  max torque per axis aka maximaler Drehmoment einer Achse
torque_max   = scale_torque * np.sqrt(3)
scale_omega  = 5        # [rad/s]  observation normalisation

#target attitude
q_desired_array_global    = np.array([1.0, 0.0, 0.0, 0.0])
omega_desired_array_global = np.array([0.0, 0.0, 0.0])     # [rad/s]

boresight_vector_in_b_global = np.array([1.0, 0.0, 0.0])   # instrument boresight in body frame Sichtlinie, Peilrichtung oder Mittelachse

time_per_step    = 0.1    # [s]
time_per_episode = 100    # [s]

# Initial attitude angle bounds (error w.r.t. desired), [deg]
angle_bound_lower = 80
angle_bound_upper = 180

# F-zone placement parameters (exponential-map method) aka Keep out Zone geometry
vector_rotation_angle1_ratio_low  = 0.5
vector_rotation_angle1_ratio_high = 0.5
vector_rotation_angle2_low  = 0.0   # [deg]
vector_rotation_angle2_high = 0.0   # [deg]

# ---------------------------------------------------------------------------
# Numba-accelerated math helpers
# ---------------------------------------------------------------------------

@njit
def sign_fun(x):
    return 1 if x >= 0 else -1

@njit
def norm_action(action):
    return np.sqrt(action[0]**2 + action[1]**2 + action[2]**2)

@njit
def normalize_quaternion(q):
    norm = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if norm > 0:
        return q / norm
    return q

@njit
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)

@njit
def quaternion_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

@njit
def sat_ode(state, inertia, inertia_inv, torque):
    state       = state.astype(np.float32)
    inertia     = inertia.astype(np.float32)
    inertia_inv = inertia_inv.astype(np.float32)
    torque      = torque.astype(np.float32)

    q_quat = state[:4] # 4D with complex numbers
    omega  = state[4:7]

    omega_cross = np.array([
        [ 0,        -omega[2],  omega[1]],
        [ omega[2],  0,        -omega[0]],
        [-omega[1],  omega[0],  0       ],
    ], np.float32)

    omega_dot = np.dot(inertia_inv, np.dot(-omega_cross, inertia) @ omega + torque)
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float32)
    q_dot = 0.5 * quaternion_multiply(q_quat, omega_quat)

    return np.concatenate((q_dot, omega_dot))

@njit
def random_unit_quat_with_angle_bound(lower_deg, upper_deg):
    e = np.random.randn(3)
    e /= np.linalg.norm(e)
    theta = np.random.uniform(lower_deg, upper_deg) * np.pi / 180
    q = np.array([
        np.cos(theta / 2),
        e[0] * np.sin(theta / 2),
        e[1] * np.sin(theta / 2),
        e[2] * np.sin(theta / 2),
    ])
    if q[0] < 0:
        q = -q
    return q

@njit
def random_angular_rate(rate_bound=0.0):
    return np.random.uniform(low=-rate_bound, high=rate_bound, size=3)

# ---------------------------------------------------------------------------
# Reward function  (Phase 1 — with KOZ penalty)
# ---------------------------------------------------------------------------

def reward_function_with_Fzone(state, action):
    if not hasattr(reward_function_with_Fzone, 'action_prev'):
        reward_function_with_Fzone.action_prev = action.copy()

    qe_0_current = state[0]
    qe_0_prev    = state[-1]

    err_phi_current = 2 * math.acos(np.clip(qe_0_current, -1.0, 1.0))
    err_phi_prev    = 2 * math.acos(np.clip(qe_0_prev,    -1.0, 1.0))

    torque        = action * scale_torque
    torque_change = np.linalg.norm(action - reward_function_with_Fzone.action_prev) * scale_torque
    reward_function_with_Fzone.action_prev = action.copy()


    # penalise keep-out zone violations: full penalty when inside (margin <= 0),
    # exponentially decaying penalty when close but still outside
    theta_margin = state[7]
    beta, alpha  = 10, 66
    if theta_margin <= 0: #violation condition
        penalty_f_zone = beta
    else:
        penalty_f_zone = beta * math.exp(-alpha * theta_margin)

    progress_penalty = 0.0 if err_phi_current <= err_phi_prev else 1.0

    reward0 = (
        math.exp(-err_phi_current / (0.14 * 2 * np.pi))
        - 0.05 * norm_action(torque) / torque_max
        - 0.005 * torque_change
        - penalty_f_zone
        - progress_penalty
    )

    if err_phi_current <= 0.25 * np.pi / 180:
        return reward0 + 9
    return reward0

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SatDynEnv(gym.Env):
    """
    Gymnasium environment for satellite attitude control with a single Keep-Out Zone.

    Observation (normalised, shape=(13,)):
        [qe(4), omega_e/scale_omega(3), theta_margin_norm(1), theta_norm(1),
         relative_avoid_vec_in_b(3), qe_0_prev(1)]

    Action (shape=(3,)):  normalised torques in [-1, 1]
    """

    def __init__(self, angle_bound_lower=80, angle_bound_upper=180,
             beta=10, alpha=66, scale_torque=2,
             time_per_episode=100, time_per_step=0.1):
        super().__init__()
        self._angle_bound_lower = angle_bound_lower  # smallest starting angle error in degrees
        self._angle_bound_upper = angle_bound_upper  # largest starting angle error in degrees
        self._beta              = beta               # how much the satellite is punished for entering the keep-out zone
        self._alpha             = alpha              # how quickly the punishment grows near the keep-out zone edge
        self._scale_torque      = scale_torque       # maximum force the thrusters can apply per axis in Nm
        # replace the globals where used:
        # angle_bound_lower → self._angle_bound_lower  (in reset)
        # beta/alpha        → self._beta / self._alpha  (in reward_function_with_Fzone)

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([
                -1, -1, -1, -1,
                -scale_omega, -scale_omega, -scale_omega,
                -np.pi / 2,
                0,
                -1, -1, -1,
                -1,
            ], dtype=np.float32),
            high=np.array([
                1, 1, 1, 1,
                scale_omega, scale_omega, scale_omega,
                np.pi,
                np.pi,
                1, 1, 1,
                1,
            ], dtype=np.float32),
        )

        self.q_desired_array   = q_desired_array_global.copy()
        self.omega_desired_array = omega_desired_array_global.copy()

        self.inertia = np.array([ #intertia tensor aka Massenträgheitsverteilung im Raum -> currently asymmetric
            [60,  5,  1],
            [ 5, 50,  2],
            [ 1,  2, 70],
        ], dtype=np.float32)

        self.dt        = time_per_step
        self.max_steps = int(time_per_episode / self.dt)
        self.steps     = 0
        self.f_zone    = None

        # Initialise state so the object is valid before reset() is called
        self.state = np.zeros(13, dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clear cross-episode reward state
        if hasattr(reward_function_with_Fzone, 'action_prev'):
            del reward_function_with_Fzone.action_prev

        q_e_initial  = random_unit_quat_with_angle_bound(angle_bound_lower, angle_bound_upper)
        omega_initial = random_angular_rate(rate_bound=1.0e-3 * np.pi / 180)

        q_abs_initial = quaternion_multiply(self.q_desired_array, q_e_initial)

        boresight_b      = boresight_vector_in_b_global.copy()
        boresight_b_quat = np.concatenate(([0.0], boresight_b))

        boresight_i_initial_quat = quaternion_multiply(
            q_abs_initial, quaternion_multiply(boresight_b_quat, quaternion_conj(q_abs_initial))
        )
        boresight_i_desired_quat = quaternion_multiply(
            self.q_desired_array, quaternion_multiply(boresight_b_quat, quaternion_conj(self.q_desired_array))
        )

        ratio1 = np.random.uniform(vector_rotation_angle1_ratio_low, vector_rotation_angle1_ratio_high)
        angle2 = np.random.uniform(vector_rotation_angle2_low, vector_rotation_angle2_high)

        avoid_vec_i, half_angle_max = generate_avoid_vector_in_i_for_1Fzone_phase1_v2(
            boresight_b,
            boresight_i_initial_quat[1:4],
            boresight_i_desired_quat[1:4],
            q_abs_initial,
            q_e_initial,
            ratio1,
            angle2,
        )

        if half_angle_max == 0.0:
            half_angle = 0.0
        else:
            half_angle_max = np.minimum(half_angle_max, 30.0)
            # KOZ size: randomly sampled between 15° and half_angle_max (capped at 30°).
            # Increase the lower bound to make the forbidden zone larger (harder),
            # decrease it to make it smaller (easier).
            half_angle = np.random.uniform(15.0, half_angle_max) * deg2rad

        self.f_zone = KeepOutZone(boresight_b, avoid_vec_i, half_angle)

        avoid_vec_i_quat = np.concatenate(([0.0], avoid_vec_i))
        avoid_vec_b_quat = quaternion_multiply(
            quaternion_conj(q_abs_initial), quaternion_multiply(avoid_vec_i_quat, q_abs_initial)
        )

        theta        = np.arccos(np.clip(np.inner(avoid_vec_b_quat[1:4], boresight_b), -1.0, 1.0))
        theta_margin = theta - half_angle

        rel_avoid_b = avoid_vec_b_quat[1:4] - boresight_b
        rel_avoid_b = rel_avoid_b / np.linalg.norm(rel_avoid_b)

        self.state = np.concatenate((
            q_e_initial, omega_initial,
            [theta_margin], [theta],
            rel_avoid_b,
            [q_e_initial[0]],
        ), dtype=np.float32)

        self.steps = 0

        return self._normalise(), {}

    # ------------------------------------------------------------------

    def step(self, action):
        action       = np.asarray(action, dtype=np.float32)
        inertia_inv  = np.linalg.inv(self.inertia)
        qe_0_prev    = self.state[0]

        # 4th-order Runge-Kutta integration
        torque = action * scale_torque
        f1 = self.dt * sat_ode(self.state[:7], self.inertia, inertia_inv, torque)
        f2 = self.dt * sat_ode(self.state[:7] + 0.5*f1, self.inertia, inertia_inv, torque)
        f3 = self.dt * sat_ode(self.state[:7] + 0.5*f2, self.inertia, inertia_inv, torque)
        f4 = self.dt * sat_ode(self.state[:7] + f3,     self.inertia, inertia_inv, torque)
        self.state[:7] += (f1 + 2*f2 + 2*f3 + f4) / 6

        self.state[:4] = normalize_quaternion(self.state[:4])

        q_abs = quaternion_multiply(self.q_desired_array, self.state[:4])

        avoid_vec_i_quat = np.concatenate(([0.0], self.f_zone.avoid_vector_in_i))
        avoid_vec_b_quat = quaternion_multiply(
            quaternion_conj(q_abs), quaternion_multiply(avoid_vec_i_quat, q_abs)
        )

        boresight_b = self.f_zone.boresight_vector_in_b
        theta        = np.arccos(np.clip(np.inner(avoid_vec_b_quat[1:4], boresight_b), -1.0, 1.0))
        theta_margin = theta - self.f_zone.half_angle

        rel_avoid_b = avoid_vec_b_quat[1:4] - boresight_b
        rel_avoid_b = rel_avoid_b / np.linalg.norm(rel_avoid_b)

        self.state[7]    = theta_margin
        self.state[8]    = theta
        self.state[9:12] = rel_avoid_b
        self.state[12]   = qe_0_prev

        reward = reward_function_with_Fzone(self.state, action)

        self.steps += 1
        done = self.steps >= self.max_steps

        return self._normalise(), reward, done, False, {}

    # ------------------------------------------------------------------

    def render(self):
        err_deg = 2 * np.degrees(math.acos(np.clip(self.state[0], -1.0, 1.0)))
        print(
            f"step={self.steps:4d}  "
            f"att_err={err_deg:.2f}deg  "
            f"omega_err={self.state[4:7]*rad2deg}  "
            f"theta_margin={self.state[7]*rad2deg:.2f}deg  "
            f"theta={self.state[8]*rad2deg:.2f}deg"
        )

    def close(self):
        pass

    # ------------------------------------------------------------------

    def _normalise(self):
        q_e_norm          = self.state[:4]              # quaternion error stays as-is, already in [-1, 1]
        omega_norm        = self.state[4:7] / scale_omega   # scale angular rate to [-1, 1] using max expected rate
        theta_margin_norm = -1 + (self.state[7] + np.pi/2) * 4 / (3*np.pi)  # maps [-π/2, π] → [-1, 1]; zero crossing at -1/3 marks KOZ boundary
        theta_norm        = -1 + self.state[8] * 2 / np.pi  # maps [0, π] → [-1, 1]
        rel_avoid_norm    = self.state[9:12]            # unit vector, already in [-1, 1]
        qe0_prev_norm     = self.state[12]              # previous scalar quaternion, already in [-1, 1]


        return np.concatenate((
            q_e_norm, omega_norm,
            [theta_margin_norm], [theta_norm],
            rel_avoid_norm, [qe0_prev_norm],
        ), dtype=np.float32)