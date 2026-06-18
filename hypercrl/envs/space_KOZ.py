# Satellite attitude control env with Keep-Out Zone (Phase 1).
# State (13,): [qe(4), omega_e(3), theta_margin(1), theta(1), rel_avoid_in_b(3), qe_0_prev(1)]
# Action (3,): normalised torques in [-1, 1], scaled by scale_torque [Nm]
#
# CBF/CLF additions (coexist with existing KOZ penalty, obs shape unchanged at 13):
#   CBF  h(x)  = theta - half_angle  (= theta_margin, already in state[7])
#   CLF  V(x)  = 1 - qe_0^2          (zero at perfect alignment qe_0 -> +-1)
#
#   CBF condition:  dh/dt >= -alpha_cbf * h(x)
#   CLF condition:  dV/dt <=  gamma_clf * V(x)   (with slack, CBF takes priority)
#
#   QP safety filter projects the agent's action onto the feasible set before
#   it is applied to the plant. Penalties for condition violations are added
#   to the reward on top of the existing exponential KOZ penalty.

import math

import cvxpy as cp
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

from .subfunctions_att_constraints import KeepOutZone
from .subfunctions_att_constraints import generate_avoid_vector_in_i_for_1Fzone_phase1_v2

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

def reward_function_with_Fzone(state, action, cbf_penalty=0.0, clf_penalty=0.0):
    # cbf_penalty / clf_penalty: computed externally in step() from the CBF/CLF
    # filter conditions; default 0.0 so nothing breaks if called without them.
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
        - cbf_penalty   # CBF violation penalty (0 when condition satisfied)
        - clf_penalty   # CLF violation penalty (0 when condition satisfied)
    )

    if err_phi_current <= 0.25 * np.pi / 180:
        return reward0 + 9
    return reward0

# --- CBF / CLF safety filter ---

class CBFCLFFilter:
    """
    QP-based safety filter that projects a proposed action u onto the set of
    actions satisfying the CBF constraint (hard) and the CLF constraint (soft).

    Barrier function  h(x)  = theta - half_angle   (= theta_margin)
    Lyapunov function V(x)  = 1 - qe_0^2

    CBF condition (discrete-time approximation):
        [h(x_next) - h(x)] / dt  >=  -alpha_cbf * h(x)

    CLF condition (discrete-time approximation):
        [V(x_next) - V(x)] / dt  <=  gamma_clf * V(x)   (with slack delta)

    The Jacobians of h and V w.r.t. the action u are derived from the
    linearised rotational dynamics omega_dot = I^{-1} * torque (dominant
    term; the gyroscopic coupling is small and handled by RK4 in step()).

    Parameters
    ----------
    inertia      : (3,3) array - satellite inertia tensor
    dt           : float       - integration timestep [s]
    scale_torque : float       - action-to-torque scaling [Nm]
    alpha_cbf    : float       - CBF decay rate  (larger -> tighter safety)
    gamma_clf    : float       - CLF decay rate  (larger -> faster convergence)
    beta_cbf     : float       - penalty weight for CBF violation in reward
    beta_clf     : float       - penalty weight for CLF violation in reward
    """

    def __init__(self, inertia, dt, scale_torque,
                 alpha_cbf=1.0, gamma_clf=0.5,
                 beta_cbf=10.0, beta_clf=10.0):
        self.inertia_inv  = np.linalg.inv(inertia).astype(np.float64)
        self.dt           = dt
        self.scale_torque = scale_torque
        self.alpha_cbf    = alpha_cbf
        self.gamma_clf    = gamma_clf
        self.beta_cbf     = beta_cbf
        self.beta_clf     = beta_clf

    def _dh_du(self, state, f_zone):
        """
        Gradient of h = theta_margin w.r.t. normalised action u (shape (3,)).

        dh/du = dh/d(avoid_b) * d(avoid_b)/d(omega) * d(omega)/du
        d(omega)/du = I^{-1} * scale_torque
        d(avoid_b)/d(omega) ~= -[avoid_b]_x * dt  (skew-symmetric cross-product matrix)
        dh/d(avoid_b) = -boresight_b / sin(theta)
        """
        q      = state[:4].astype(np.float64)
        theta  = float(state[8])

        boresight_b = f_zone.boresight_vector_in_b.astype(np.float64)
        avoid_vec_i = f_zone.avoid_vector_in_i.astype(np.float64)

        q_abs   = self._quat_mul(np.array([1., 0., 0., 0.]), q)  # q_desired = [1,0,0,0]
        avoid_b = self._rotate_vec(avoid_vec_i, self._quat_conj(q_abs))

        sin_theta = math.sin(theta)
        if abs(sin_theta) < 1e-6:
            # singularity (theta ~ 0 or pi): gradient ill-defined -> zero
            return np.zeros(3)

        dh_davoid     = -boresight_b / sin_theta
        skew_avoid    = self._skew(avoid_b)
        davoid_domega = -skew_avoid * self.dt
        domega_du     = self.inertia_inv * self.scale_torque

        dh_du = dh_davoid @ davoid_domega @ domega_du
        return dh_du

    def _dV_du(self, state):
        """
        Gradient of V = 1 - qe_0^2 w.r.t. normalised action u (shape (3,)).

        dV/d(qe_0) = -2 * qe_0
        d(qe_0)/d(omega) = -0.5 * qe_1:3   (quaternion kinematics scalar part)
        d(omega)/du = I^{-1} * scale_torque
        """
        qe_0 = float(state[0])
        qe_v = state[1:4].astype(np.float64)

        dV_dqe0     = -2.0 * qe_0
        dqe0_domega = -0.5 * qe_v * self.dt
        domega_du   = self.inertia_inv * self.scale_torque

        dV_du = dV_dqe0 * (dqe0_domega @ domega_du)
        return dV_du

    def filter_action(self, u_proposed, state, f_zone):
        """
        Project u_proposed onto the CBF-safe, CLF-convergent feasible set.

        Returns
        -------
        u_safe    : (3,) corrected action in [-1, 1]
        cbf_viol  : float >= 0, magnitude of CBF condition violation BEFORE filtering
        clf_viol  : float >= 0, magnitude of CLF condition violation BEFORE filtering
        """
        u_proposed = np.asarray(u_proposed, dtype=np.float64)

        h_val = float(state[7])           # theta_margin
        V_val = 1.0 - float(state[0])**2  # CLF value

        dh = self._dh_du(state, f_zone)
        dV = self._dV_du(state)

        cbf_rhs = -self.alpha_cbf * h_val
        clf_rhs =  self.gamma_clf * V_val

        # measure violations BEFORE filtering (for reward penalties)
        cbf_residual = float(dh @ u_proposed) - cbf_rhs
        clf_residual = float(dV @ u_proposed) - clf_rhs
        cbf_viol = self.beta_cbf * max(0.0, -cbf_residual)
        clf_viol = self.beta_clf * max(0.0,  clf_residual)

        u_var = cp.Variable(3)
        delta = cp.Variable(1, nonneg=True)   # CLF slack

        objective = cp.Minimize(cp.sum_squares(u_var - u_proposed))
        constraints = [
            dh @ u_var >= cbf_rhs,             # CBF: hard safety constraint
            dV @ u_var <= clf_rhs + delta,      # CLF: soft convergence constraint
            u_var >= -1.0,
            u_var <=  1.0,
        ]

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, warm_starting=True, verbose=False,
                       eps_abs=1e-4, eps_rel=1e-4, max_iter=4000)
        except cp.SolverError:
            return u_proposed.astype(np.float32), cbf_viol, clf_viol

        if u_var.value is None:
            return u_proposed.astype(np.float32), cbf_viol, clf_viol

        u_safe = np.clip(u_var.value, -1.0, 1.0).astype(np.float32)
        return u_safe, cbf_viol, clf_viol

    @staticmethod
    def _skew(v):
        """3x3 skew-symmetric matrix for cross product: skew(v) @ w = v x w."""
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ],
        ], dtype=np.float64)

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dtype=np.float64)

    @staticmethod
    def _quat_conj(q):
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

    @staticmethod
    def _rotate_vec(v, q):
        """Rotate 3-vector v by unit quaternion q: v' = q (x) [0;v] (x) q*."""
        q = q.astype(np.float64)
        v_quat = np.array([0., v[0], v[1], v[2]], dtype=np.float64)
        res = CBFCLFFilter._quat_mul(
            CBFCLFFilter._quat_mul(q, v_quat),
            CBFCLFFilter._quat_conj(q)
        )
        return res[1:4]

# --- environment ---

class SatDynEnv(gym.Env):
    # Satellite attitude control with a single KOZ, CBF safety filter, and
    # CLF convergence enforcement. See module header for state/action layout.

    def __init__(self, angle_bound_lower=80, angle_bound_upper=180,
             beta=10, alpha=66, scale_torque=2,
             time_per_episode=100, time_per_step=0.1, inertia=None,
             alpha_cbf=1.0, gamma_clf=0.5,
             beta_cbf=10.0, beta_clf=10.0):
        super().__init__()
        self._angle_bound_lower = angle_bound_lower  # initial attitude error range [deg]
        self._angle_bound_upper = angle_bound_upper
        self._beta              = beta               # KOZ violation penalty magnitude
        self._alpha             = alpha              # KOZ penalty decay rate near boundary
        self._scale_torque      = scale_torque       # max thruster torque per axis [Nm]

        # CBF / CLF parameters
        self._alpha_cbf = alpha_cbf   # CBF decay rate
        self._gamma_clf = gamma_clf   # CLF decay rate
        self._beta_cbf  = beta_cbf    # CBF violation penalty weight
        self._beta_clf  = beta_clf    # CLF violation penalty weight

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

        # CBF/CLF filter (built after self.inertia is resolved above)
        self.cbf_clf_filter = CBFCLFFilter(
            inertia      = self.inertia,
            dt           = self.dt,
            scale_torque = self._scale_torque,
            alpha_cbf    = self._alpha_cbf,
            gamma_clf    = self._gamma_clf,
            beta_cbf     = self._beta_cbf,
            beta_clf     = self._beta_clf,
        )

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
            half_angle_max = np.minimum(half_angle_max, 30.0)
            # sample KOZ size in [15°, min(half_angle_max, 30°)]; raise lower bound = harder task
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


    def step(self, action):
        action       = np.asarray(action, dtype=np.float32)
        inertia_inv  = np.linalg.inv(self.inertia)
        qe_0_prev    = self.state[0]

        # CBF/CLF QP filter: measure pre-filter violations (for reward
        # penalties) and correct the action so it satisfies the CBF
        # constraint before integration.
        action_safe, cbf_penalty, clf_penalty = self.cbf_clf_filter.filter_action(
            action, self.state, self.f_zone
        )

        # RK4 integration (uses the safety-filtered action)
        torque = action_safe * scale_torque
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

        reward = reward_function_with_Fzone(
            self.state, action_safe,
            cbf_penalty=cbf_penalty,
            clf_penalty=clf_penalty,
        )

        self.steps += 1
        done = self.steps >= self.max_steps

        return self._normalise(), reward, done, False, {}


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


# --- validation environment (held-out, NOT used for training) ---
#
# The agent is never trained here. It exists purely to check whether a
# trained policy generalises to a larger Keep-Out Zone it has never seen.
#
# The ONLY behavioural difference from SatDynEnv is the half_angle sampling
# range in reset():
#     Training   : uniform(15°, min(half_angle_max, 30°))
#     Validation : uniform(30°, min(half_angle_max, 60°))
#
# Everything else (state shape, action shape, dynamics, reward, CBF/CLF
# filter, inertia handling, rendering) is inherited unchanged from SatDynEnv.

VAL_KOZ_MIN_DEG = 30.0   # starts exactly where training's upper bound ends
VAL_KOZ_MAX_DEG = 60.0   # unseen upper bound


class SatDynEnvValidation(SatDynEnv):
    """
    Held-out validation environment. Subclasses SatDynEnv and overrides only
    the KOZ half_angle sampling in reset(); everything else (dynamics,
    reward, CBF/CLF filter, inertia handling, rendering) is reused as-is.
    """

    def __init__(self, *args, val_koz_min_deg=VAL_KOZ_MIN_DEG,
                 val_koz_max_deg=VAL_KOZ_MAX_DEG, **kwargs):
        self._val_koz_min_deg = val_koz_min_deg
        self._val_koz_max_deg = val_koz_max_deg
        self._current_half_angle_deg = 0.0
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        # Run gymnasium's seeding first (mirrors SatDynEnv.reset() behaviour)
        gym.Env.reset(self, seed=seed)

        if hasattr(reward_function_with_Fzone, 'action_prev'):
            del reward_function_with_Fzone.action_prev

        q_e_initial   = random_unit_quat_with_angle_bound(angle_bound_lower, angle_bound_upper)
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
            self._current_half_angle_deg = 0.0
        else:
            # KEY DIFFERENCE FROM TRAINING ENV (SatDynEnv.reset):
            # Training   : uniform(15°, min(half_angle_max, 30°))
            # Validation : uniform(30°, min(half_angle_max, 60°))
            val_max = np.minimum(half_angle_max, self._val_koz_max_deg)

            if val_max <= self._val_koz_min_deg:
                # geometry too tight to place a large KOZ - use whatever fits
                half_angle = val_max * deg2rad
            else:
                half_angle = np.random.uniform(self._val_koz_min_deg, val_max) * deg2rad

            self._current_half_angle_deg = half_angle * rad2deg

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
        # Reuse SatDynEnv.step() exactly, then attach validation-only logging
        # (KOZ size this episode, violation flag) to the info dict.
        obs, reward, done, truncated, info = super().step(action)
        info["half_angle_deg"] = self._current_half_angle_deg
        info["koz_violation"]  = bool(self.state[7] <= 0)
        return obs, reward, done, truncated, info

    def render(self):
        err_deg = 2 * np.degrees(math.acos(np.clip(self.state[0], -1.0, 1.0)))
        print(
            f"[VAL] step={self.steps:4d}  "
            f"att_err={err_deg:.2f}deg  "
            f"theta_margin={self.state[7]*rad2deg:.2f}deg  "
            f"KOZ_half={self._current_half_angle_deg:.1f}deg"
        )
        self._pyvista_render()