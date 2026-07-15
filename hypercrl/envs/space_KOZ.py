# Satellite attitude control env with Keep-Out Zone (Phase 1).
# State (13,): [qe(4), omega_e(3), theta_margin(1), theta(1), rel_avoid_in_b(3), qe_0_prev(1)]
# Action (3,): normalised torques in [-1, 1], scaled by scale_torque [Nm]

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

from .subfunctions_att_constraints import KeepOutZone
from .subfunctions_att_constraints import generate_avoid_vector_in_i_for_1Fzone_phase1_v2
from .space_cbf_clf import SpaceAttitudeCBF, SpaceAttitudeCLF
from hypercrl.control.safety_filter import SafetyFilter

deg2rad = np.pi / 180
rad2deg = 180 / np.pi

scale_torque = 2        # [Nm] max torque per axis
torque_max   = scale_torque * np.sqrt(3)
scale_omega  = 5        # [rad/s] used to normalise observations

q_desired_array_global    = np.array([1.0, 0.0, 0.0, 0.0])   # target attitude (identity)
omega_desired_array_global = np.array([0.0, 0.0, 0.0])        # target angular rate
boresight_vector_in_b_global = np.array([1.0, 0.0, 0.0])      # instrument axis in body frame (+X)

time_per_step    = 0.1    # [s]
time_per_episode = 100    # [s]

# initial attitude error bounds [deg]
angle_bound_lower = 80
angle_bound_upper = 180

# KOZ half-angle sampling bounds [deg] (upper is additionally capped by the
# geometry-dependent half_angle_max computed at reset)
half_angle_low_deg = 15.0
half_angle_high_deg = 30.0

# KOZ placement parameters (exponential-map method)
vector_rotation_angle1_ratio_low  = 0.5
vector_rotation_angle1_ratio_high = 0.5
vector_rotation_angle2_low  = 0.0   # [deg]
vector_rotation_angle2_high = 0.0   # [deg]

# --- math helpers (Numba JIT) ---

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

    q_quat = state[:4]
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

# --- reward function ---

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


    # full penalty inside KOZ, exponential decay outside
    theta_margin = state[7]
    beta, alpha  = 10, 66
    if theta_margin <= 0:
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

# --- environment ---

class SatDynEnv(gym.Env):
    # Satellite attitude control with a single KOZ. See module header for state/action layout.

    def __init__(self, angle_bound_lower=80, angle_bound_upper=180,
             beta=10, alpha=66, scale_torque=2,
             time_per_episode=100, time_per_step=0.1, inertia=None):
        super().__init__()
        self._angle_bound_lower = angle_bound_lower  # initial attitude error range [deg]
        self._angle_bound_upper = angle_bound_upper
        self._beta              = beta               # KOZ violation penalty magnitude
        self._alpha             = alpha              # KOZ penalty decay rate near boundary
        self._scale_torque      = scale_torque       # max thruster torque per axis [Nm]

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # All 13 components are normalised to [-1, 1] by _normalise():
        #   [0:4]  q_e          — unit-quaternion components
        #   [4:7]  omega/5      — angular rate divided by scale_omega
        #   [7]    theta_margin_norm = -1 + (margin + π/2)*4/(3π)
        #   [8]    theta_norm   = -1 + theta*2/π
        #   [9:12] rel_avoid_b  — unit-vector difference components
        #   [12]   qe_0_prev    — previous scalar quaternion component
        self.observation_space = spaces.Box(
            low=-np.ones(13, dtype=np.float32),
            high=np.ones(13, dtype=np.float32),
        )

        self.q_desired_array   = q_desired_array_global.copy()
        self.omega_desired_array = omega_desired_array_global.copy()

        if inertia is None: # intertia tensor aka Massenträgheitsverteilung -> default asymmetric
            self.inertia = np.array([
                [60,  5,  1],
                [ 5, 50,  2],
                [ 1,  2, 70],
            ], dtype=np.float32)
        else:
            self.inertia = np.array(inertia, dtype=np.float32)

        self.dt        = time_per_step
        self.max_steps = int(time_per_episode / self.dt)
        self.steps     = 0
        self.f_zone    = None

        self.state = np.zeros(13, dtype=np.float32)  # placeholder until reset() runs
        self.reset()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # clear stateful action memory from previous episode
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
            half_angle_max = np.minimum(half_angle_max, half_angle_high_deg)
            # sample KOZ size in [low, min(half_angle_max, high)] (module
            # globals; a fixed-scenario run pins low == high).  Raising the
            # lower bound = harder task.
            lo = np.minimum(half_angle_low_deg, half_angle_max)
            half_angle = np.random.uniform(lo, half_angle_max) * deg2rad

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


    def step(self, action):
        action       = np.asarray(action, dtype=np.float32)
        inertia_inv  = np.linalg.inv(self.inertia)
        qe_0_prev    = self.state[0]

        # RK4 integration
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

        return self._normalise(), reward, done, False, {"keep_out_violation": bool(theta_margin < 0)}


    def render(self):
        err_deg = 2 * np.degrees(math.acos(np.clip(self.state[0], -1.0, 1.0)))
        print(
            f"step={self.steps:4d}  "
            f"att_err={err_deg:.2f}deg  "
            f"omega_err={self.state[4:7]*rad2deg}  "
            f"theta_margin={self.state[7]*rad2deg:.2f}deg  "
            f"theta={self.state[8]*rad2deg:.2f}deg"
        )
        self._pyvista_render()

    def _pyvista_render(self):
        try:
            import pyvista as pv
            from scipy.spatial.transform import Rotation
        except ImportError:
            return

        # First call: build static scene elements once, add dynamic placeholders.
        if not hasattr(self, '_pl') or self._pl is None:
            # --- static elements (task geometry, never change) ---
            koz_cone = pv.Cone(
                center=self.f_zone.avoid_vector_in_i * 0.45,
                direction=-self.f_zone.avoid_vector_in_i,
                angle=np.degrees(self.f_zone.half_angle),
                height=0.9,
                resolution=80,
            )
            forbidden = pv.PlatonicSolid('dodecahedron')
            forbidden.scale([0.1, 0.1, 0.1], inplace=True)
            forbidden.translate(self.f_zone.avoid_vector_in_i * 0.9, inplace=True)

            r_des = Rotation.from_quat([
                self.q_desired_array[1], self.q_desired_array[2],
                self.q_desired_array[3], self.q_desired_array[0],
            ])
            goal_dir   = r_des.as_matrix() @ self.f_zone.boresight_vector_in_b
            goal_arrow = pv.Arrow(start=[0, 0, 0], direction=goal_dir, scale=0.3)

            # --- dynamic placeholders (updated every frame) ---
            pv.global_theme.allow_empty_mesh = True
            self._sat_mesh  = pv.PolyData()
            self._bore_mesh = pv.PolyData()

            self._pl = pv.Plotter()
            self._pl.add_axes()
            self._pl.add_mesh(self._sat_mesh,  color='silver', label='Satellite')
            self._pl.add_mesh(self._bore_mesh, color='green',  label='Boresight (current)')
            self._pl.add_mesh(koz_cone,        color='red',    opacity=0.6, label='KOZ')
            self._pl.add_mesh(forbidden,       color='black',  label='Forbidden object')
            self._pl.add_mesh(goal_arrow,      color='yellow', label='Goal')
            self._pl.add_legend()
            self._pl.show(interactive_update=True, title="SatDynEnv live render")

        # Build dynamic geometry for the current attitude
        q_abs = quaternion_multiply(self.q_desired_array, self.state[:4])
        r = Rotation.from_quat([q_abs[1], q_abs[2], q_abs[3], q_abs[0]])
        T = np.eye(4)
        T[:3, :3] = r.as_matrix()
        R = T[:3, :3]

        body    = pv.Box(bounds=(-0.06,  0.06, -0.06,  0.06, -0.09,  0.09))
        panel_l = pv.Box(bounds=(-0.08,  0.08, -0.28, -0.07, -0.005, 0.005))
        panel_r = pv.Box(bounds=(-0.08,  0.08,  0.07,  0.28, -0.005, 0.005))
        sat = pv.merge([body, panel_l, panel_r])
        sat.transform(T, inplace=True)

        boresight_i     = R @ self.f_zone.boresight_vector_in_b
        boresight_arrow = pv.Arrow(start=[0, 0, 0], direction=boresight_i, scale=0.2)

        self._sat_mesh.copy_from(sat)
        self._bore_mesh.copy_from(boresight_arrow)
        self._pl.update()

    # ------------------------------------------------------------------
    # Safety filter interface (paper Eq. 19)
    # ------------------------------------------------------------------

    def get_cbf(self, gamma: float = 0.5) -> SpaceAttitudeCBF:
        return SpaceAttitudeCBF(self, gamma=gamma)

    def get_clf(
        self,
        c_q: float = 1.0,
        c_w: float = 0.1,
        zeta_min: float = 0.001,
        zeta_max: float = 0.06,
        j: float = 5.0,
        c: float = 0.6,
    ) -> SpaceAttitudeCLF:
        return SpaceAttitudeCLF(self, c_q=c_q, c_w=c_w, zeta_min=zeta_min,
                                zeta_max=zeta_max, j=j, c=c)

    def get_safety_filter(
        self,
        # ε > 0 keeps a margin off the safe-set boundary to cover the dt=0.1
        # discretization gap (paper Sec. IV-A uses ε = 0.01): with ε = 0 a
        # single discrete step can tunnel across H = 0 before the one-step QP
        # can react.
        cbf_epsilon: float = 0.01,
        clf_rho: float = 0.001,
        gamma: float = 0.5,
    ) -> SafetyFilter:
        return SafetyFilter(
            cbf=self.get_cbf(gamma=gamma),
            clf=self.get_clf(),
            u_max=1.0,           # action space is [-1,1]³
            control_dim=3,
            cbf_epsilon=cbf_epsilon,
            clf_rho=clf_rho,
        )

    def close(self):
        if hasattr(self, '_pl') and self._pl is not None:
            self._pl.close()
            self._pl = None
            self._sat_mesh = self._bore_mesh = self._koz_mesh = None
            self._forb_mesh = self._goal_mesh = None


    def _normalise(self):
        q_e_norm          = self.state[:4]
        omega_norm        = self.state[4:7] / scale_omega
        theta_margin_norm = -1 + (self.state[7] + np.pi/2) * 4 / (3*np.pi)  # [-π/2, π] → [-1, 1]
        theta_norm        = -1 + self.state[8] * 2 / np.pi                   # [0, π]    → [-1, 1]
        rel_avoid_norm    = self.state[9:12]
        qe0_prev_norm     = self.state[12]


        return np.concatenate((
            q_e_norm, omega_norm,
            [theta_margin_norm], [theta_norm],
            rel_avoid_norm, [qe0_prev_norm],
        ), dtype=np.float32)