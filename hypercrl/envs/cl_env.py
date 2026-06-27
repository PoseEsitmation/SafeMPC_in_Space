import gymnasium as gym
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium.wrappers import TimeLimit
from hypercrl.envs.mujoco.half_cheetah_safe import HalfCheetahSafeEnv


# robosuite
try:
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    from robosuite.wrappers import GymWrapper
    _ROBOSUITE_IMPORT_ERROR = None
except Exception as err:
    suite = None
    load_controller_config = None
    GymWrapper = None
    _ROBOSUITE_IMPORT_ERROR = err

# Custom Env
from .lqr import LQR_2DCar, LQR_HARD

Rots = [[0, 0, 0], [0, 10, 0], [0, 20, 0], [0, 30, 0],
        [-10, -10, 0], [-10, -20, 0], [-10, 30, 0],
        [15, -5, 25], [-30, -30, -5], [-20, 20, -20],
        [0, -20, 10]]

CHEETAH_ENVS = ['MBRLHalfCheetah-v0', 'HalfCheetahBigTorso-v0', 'HalfCheetahBigThigh-v0',
                'HalfCheetahBigLeg-v0', 'HalfCheetahBigFoot-v0']
HALF_CHEETAH_SAFE_ENVS = [
    lambda render_mode=None: HalfCheetahSafeEnv(keep_out_zones=[(4.0, 4.5)], render_mode=render_mode),   # Task 0
    lambda render_mode=None: HalfCheetahSafeEnv(keep_out_zones=[(8.0, 8.5)], render_mode=render_mode),   # Task 1
    lambda render_mode=None: HalfCheetahSafeEnv(keep_out_zones=[(12.0, 12.5)], render_mode=render_mode), # Task 2
]
WALKER_ENVS = ['MBRLWalker-v0', 'Walker2dBigTorso-v0', 'Walker2dBigThigh-v0',
               'Walker2dBigLeg-v0', 'Walker2dBigFoot-v0']
HOPPER_ENVS = ['MBRLHopper-v0', 'HopperBigTorso-v0', 'HopperBigThigh-v0',
               'HopperBigLeg-v0', 'HopperBigFoot-v0']
INVERTED_PENDULUM_ENVS = ['InvertedPendulum-v2', 'InvertedPendulumSmallPole-v0',
                          'InvertedPendulumBigPole-v0']
INVERTED_PENDULUM_BIN_ENVS = [0, 2, -2, 4, -4]
CARTPOLE_ENVS = ['MBRLCartpole-v0', 'CartpoleLong1-v0', 'CartpoleShort1-v0',
                 'CartpoleLong2-v0', 'CartpoleLong3-v0', 'CartpoleLong4-v0',
                 'CartpoleShort2-v0', 'CartpoleLong5-v0', 'CartpoleLong6-v0',
                 'CartpoleLong7-v0']
CARTPOLE_BIN_ENVS = ['MBRLCartpole-v0',
                     'CartpoleLeft1-v0', 'CartpoleRight1-v0']
REACHER_ENVS = ['Reacher-v2', 'ReacherShort1-v0', 'ReacherLong1-v0', 'ReacherShort2-v0',
                'ReacherLong2-v0', 'ReacherShort3-v0', 'ReacherLong3-v0',
                'ReacherShort4-v0', 'ReacherLong4-v0', 'ReacherShort5-v0']

PUSH_ENV = [[500, 500], [100, 500], [500, 100], [500, 250], [250, 500],
            [1000, 100], [100, 1000], [300, 1000], [1000, 300], [1000, 1000]]
# PUSH_ENV = [[100, 100], [20, 100], [100, 20], [100, 50], [50, 100],
#             [1000, 100], [100, 1000], [300, 1000], [1000, 300], [1000, 1000]]
DOOR_ENV = [("pull", [-1.57, 1.57]), ("round", [-1.57, 0.]), ("lever", [-1.57, 0]),
            ("round", [0., 1.57]), ("lever", [0., 1.57])]

ROTATE_ENV = [[[0., -0.02, 0.84029956], [0.70710678118, 0, 0, 0.70710678118],  # T1 pos1, quat1,
               [0., 0.02, 0.84029956], [0.70710678118, 0, 0, 0.70710678118]],  # .  pos2, quat2
              [[0., -0.02, 0.84029956], [0.70710678118, 0, 0, -0.70710678118],   # T2
               [0., 0.02, 0.84029956], [0.70710678118, 0, 0, -0.70710678118]],
              [[0.02, 0, 0.84029956], [0, 0, 0, 1],   # T3
               [-0.02, 0, 0.84029956], [0, 0, 0, 1]],
              [[-0.02, 0, 0.84029956], [0, 0, 0, 0],   # T4
               [0.04, 0, 0.84029956], [0.70710678118, 0, 0, -0.70710678118]],
              [[-0.04, 0., 0.84029956], [0.70710678118, 0, 0, 0.70710678118],   # T5
               [0.02, 0., 0.84029956], [0, 0, 0, 0]]]
SLIDE_ENV = [0.001, 0.0005, 0.002, 0.0026, 0.005]

SPACE_ENV_PRESETS = [
    {},                                                                                    # Task 0 — default: large starting error (80–180°), full torque, standard KOZ penalty
    {"angle_bound_lower": 10,  "angle_bound_upper": 45},                                  # Task 1 — easy: small starting error (10–45°)
    {"angle_bound_lower": 90,  "angle_bound_upper": 180, "beta": 50, "alpha": 100},       # Task 2 — hard: large starting error + 5x stronger KOZ penalty
    {"scale_torque": 0.5},                                                                 # Task 3 — weak: half thruster power (0.5 Nm)
]

SPACE_MOI_ENVS = [
    {"inertia": [[60, 5, 1], [5, 50, 2], [1, 2, 70]]},       # Task 0 — baseline asymmetric (current default)
    {"inertia": [[20, 1, 0], [1, 22, 0], [0, 0, 25]]},        # Task 1 — nearly symmetric, small satellite
    {"inertia": [[120, 10, 3], [10, 90, 5], [3, 5, 150]]},    # Task 2 — heavy asymmetric, large satellite
    {"inertia": [[80, 2, 0], [2, 80, 0], [0, 0, 20]]},        # Task 3 — oblate (flat disk shape)
]




class EnvSpecs():
    unit = {
        "pusher": ["m", "m", "m", "m", "m", "m", "m", "m", "m", "m"],
        "door": ["m", "m", "m", "rad"],
        "door_pose": ["m", "m", "m", "", "", "", "", "m",
                      "", "", "", "", "", "", "", "", "", "", "", "",
                      "", "", "", "", "rad", "rad"]
    }
    names = {
        "pusher": ["x", "y", "x_c1", "y_c1", "x_c2", "y_c2",
                   "x_c3", "y_c3", "x_c4", "y_c4"],
        "door": ["x", "y", "z", "joint_pos"],
        "door_pose": ["h_X_ee", "h_Y_ee", "h_Z_ee", "h_Qw_ee", "h_Qx_ee",
                      "h_Qy_ee", "h_Qz_ee", "grip", "", "", "", "", "", "", "", "",
                      "", "", "", "", "", "", "knob_vel", "door_vel", "knob_ang", "door_ang"]
    }

    a_dims = {
        "pusher": 2,
        'pusher_rot': 2,
        "pusher_slide": 2,
        "reacher": 2,
        "half_cheetah": 6,
        "half_cheetah_body": 6,
        "half_cheetah_safe": 6,
        "cartpole": 1,
        "cartpole_bin": 1,
        "inverted_pendulum": 1,
        "lqr": 4,
        "lqr10": 20,
        "door": 3,
        "door_pose": 7,
        "spaceEnv": 3,
        "spaceEnv_moi": 3,
    }

    x_dims = {
        "pusher": 10,
        "pusher_rot": 20,
        "pusher_slide": 18,
        "reacher": 11,
        "half_cheetah": 18,
        "half_cheetah_body": 18,
        "half_cheetah_safe": 19,
        "cartpole": 4,
        "cartpole_bin": 4,
        "inverted_pendulum": 4,
        "lqr": 4,
        "lqr10": 20,
        "door": 4,
        "door_pose": 10,
        "spaceEnv": 13,
        "spaceEnv_moi": 13,
    }

    @classmethod
    def get_dim_unit(cls, env):
        if env in cls.unit:
            return cls.unit[env]
        else:
            return [" " for i in range(cls.x_dims[env])]

    @classmethod
    def get_dim_name(cls, env):
        if env in cls.names:
            return cls.names[env]
        else:
            return [f"Dim {i+1}" for i in range(cls.x_dims[env])]


class CLEnvHandler():
    def __init__(self, env, seed):
        self.cl_env = env
        self.seed = seed

        self._envs = []
        self._env_mt_world = None

    def add_task(self, task_id, render=False, replica=False):
        if self.cl_env in {"pusher", "pusher_rot", "pusher_slide", "door", "door_pose"} and suite is None:
            raise ImportError(
                "robosuite-based environments require robosuite + mujoco-py. "
                "On Python 3.12, use non-robosuite environments or a Python version that supports mujoco-py."
            ) from _ROBOSUITE_IMPORT_ERROR

        # Meta world environemnt has its own wrapper
        if self.cl_env.startswith("metaworld"):
            if self._env_mt_world is None:
                # External Env
                from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
                from metaworld.envs.mujoco.env_dict import (EASY_MODE_CLS_DICT,
                                                            EASY_MODE_ARGS_KWARGS)
                env = MultiClassMultiTaskEnv(
                    task_env_cls_dict=EASY_MODE_CLS_DICT,
                    task_args_kwargs=EASY_MODE_ARGS_KWARGS,
                    sample_goals=False,
                    obs_type='plain',
                )
                goals_dict = {
                    t: [e.goal.copy()]
                    for t, e in zip(env._task_names, env._task_envs)
                }
                env.discretize_goal_space(goals_dict)
                self._env_mt_world = env
            return self.get_env(task_id)

        #assert task_id <= len(self._envs) removing since we want an untrained task number which is less then the one already trained.
        if self.cl_env == "lqr":
            env = LQR_2DCar(friction=0.5 * task_id)
        elif self.cl_env == "lqr10":
            env = LQR_HARD(friction=0.5 * task_id)
        elif self.cl_env == "pendulum":
            env = gym.make('Pendulum-v0')
            env.env.g = 10 + task_id * 2
        elif self.cl_env == "pendulum2":
            gravity = [10, 0, 2, 8, 12, 5, 4, 14, 9, 7, 30]
            env = gym.make('Pendulum-v0')
            env.env.g = gravity[task_id]
        elif self.cl_env == "humanoid":
            env = gym.make('Humanoid-v4',
                           render_mode="human" if render else None)
            rot = R.from_euler('zxz', Rots[task_id], degrees=True)
            g = rot.apply(np.array([0, 0, -9.81]))
            env.model.opt.gravity[:] = g
            print(env.model.opt.gravity)
        elif self.cl_env == "hopper_body":
            env = gym.make(HOPPER_ENVS[task_id])
        elif self.cl_env == "walker_body":
            env = gym.make(WALKER_ENVS[task_id])
        elif self.cl_env == "inverted_pendulum":
            env = gym.make(INVERTED_PENDULUM_ENVS[task_id])
        elif self.cl_env == "half_cheetah_body":
            env = gym.make(CHEETAH_ENVS[task_id],
                           render_mode="human" if render else None)
        elif self.cl_env == "half_cheetah_safe":
            env = TimeLimit(
                HALF_CHEETAH_SAFE_ENVS[task_id](render_mode="human" if render else None),
                max_episode_steps=1000,
            )
        elif self.cl_env == "half_cheetah":
            env = gym.make('MBRLHalfCheetah-v0',
                           render_mode="human" if render else None)
            rot = R.from_euler('zxz', Rots[task_id], degrees=True)
            g = rot.apply(np.array([0, 0, -9.81]))
            env.model.opt.gravity[:] = g
            print(env.model.opt.gravity)
        elif self.cl_env == "inverted_pendulum_bin":
            from .mujoco.modified_invertedpendulum import InvertedPendulumBin
            env = TimeLimit(InvertedPendulumBin(
                INVERTED_PENDULUM_BIN_ENVS[task_id]), 1000)
        elif self.cl_env == "cartpole_bin":
            from .cartpole import CartpoleBinEnv
            env = gym.make(
                CARTPOLE_BIN_ENVS[task_id], render_mode="human" if render else None)
        elif self.cl_env == "cartpole":
            env = gym.make(CARTPOLE_ENVS[task_id],
                           render_mode="human" if render else None)
        elif self.cl_env == "reacher":
            env = gym.make(REACHER_ENVS[task_id])
        elif self.cl_env == "pusher":
            from .rs import PandaCL
            env = suite.make(env_name="PandaCL", density=PUSH_ENV[task_id], robots="Panda",
                             controller_configs=load_controller_config(
                                 default_controller="OSC_POSITION"),
                             has_renderer=render)
            env = GymWrapper(env)
        # For openai GYM environments, we set seed and wrap with monitor
        elif self.cl_env == "pusher_rot":
            from .rs import PandaRot
            env = suite.make(env_name="PandaRot", robots="Panda", start_poses=ROTATE_ENV[task_id],
                             controller_configs=load_controller_config(
                                 default_controller="OSC_POSITION"),
                             has_renderer=render)
            env = GymWrapper(env)
        elif self.cl_env == "pusher_slide":
            from .rs import PandaSlide
            env = suite.make(env_name="PandaSlide", robots="Panda", box2_friction=SLIDE_ENV[task_id],
                             controller_configs=load_controller_config(
                                 default_controller="OSC_POSITION"),
                             has_renderer=render)
            env = GymWrapper(env)
        elif self.cl_env == "door":
            from .rs import PandaDoor
            env = suite.make(env_name="PandaDoor", handle_type="pull", robots="Panda",
                             controller_configs=load_controller_config(
                                 default_controller="OSC_POSITION"),
                             has_renderer=render)
            env = GymWrapper(env)
        elif self.cl_env == "door_pose":
            from .rs import PandaDoor
            env = suite.make(env_name="PandaDoor", handle_type=DOOR_ENV[task_id][0],
                             joint_range=DOOR_ENV[task_id][1], robots="Panda",
                             controller_configs=load_controller_config(
                                 default_controller="OSC_POSE"),
                             pose_control=True, has_renderer=render)
            env = GymWrapper(env)
        elif self.cl_env == "spaceEnv_moi":
            from .space_KOZ import SatDynEnv
            env = SatDynEnv(**SPACE_MOI_ENVS[task_id],
                            render_mode="human" if render else None)
        elif self.cl_env == "spaceEnv":
            from .space_KOZ import SatDynEnv
            env = SatDynEnv(**SPACE_ENV_PRESETS[task_id],
                            render_mode="human" if render else None)
        if not self.cl_env.startswith("lqr"):
            if hasattr(env, 'seed'):
                env.seed(self.seed)



        if not replica:
            self._envs.append(env)
            return self.get_env(task_id)
        else:
            return env

    def get_env(self, task_id):

        if self.cl_env == "metaworld10":
            assert task_id < 10 and task_id >= 0
            self._env_mt_world.set_task(0)
            return TimeLimit(self._env_mt_world, 150)

        assert task_id < len(self._envs) and task_id >= 0
        return self._envs[task_id]

    def close(self):
        if self.cl_env == "metaworld10":
            self._env_mt_world.close()
            return

        for env in self._envs:
            env.close()

    @staticmethod
    def describe_task(env_name: str, task_id: int) -> dict:
        """Return a serialisable description of a task for the tasks.json manifest."""
        desc: dict = {"task_id": task_id, "env": env_name}
        if env_name == "cartpole":
            desc["gym_id"] = CARTPOLE_ENVS[task_id] if task_id < len(CARTPOLE_ENVS) else None
        elif env_name == "cartpole_bin":
            desc["gym_id"] = CARTPOLE_BIN_ENVS[task_id] if task_id < len(CARTPOLE_BIN_ENVS) else None
        elif env_name in ("lqr", "lqr10"):
            desc["friction"] = 0.5 * task_id
        elif env_name in ("half_cheetah_body", "half_cheetah"):
            desc["gym_id"] = CHEETAH_ENVS[task_id] if task_id < len(CHEETAH_ENVS) else None
        elif env_name == "half_cheetah_safe":
            desc["keep_out_zones"] = str(HALF_CHEETAH_SAFE_ENVS[task_id]) if task_id < len(HALF_CHEETAH_SAFE_ENVS) else None
        elif env_name == "inverted_pendulum":
            desc["gym_id"] = INVERTED_PENDULUM_ENVS[task_id] if task_id < len(INVERTED_PENDULUM_ENVS) else None
        elif env_name == "reacher":
            desc["gym_id"] = REACHER_ENVS[task_id] if task_id < len(REACHER_ENVS) else None
        elif env_name == "spaceEnv_moi":
            desc["params"] = SPACE_MOI_ENVS[task_id] if task_id < len(SPACE_MOI_ENVS) else {}
        elif env_name.startswith("spaceEnv"):
            desc["params"] = SPACE_ENV_PRESETS[task_id] if task_id < len(SPACE_ENV_PRESETS) else {}
        elif env_name == "door_pose":
            desc["handle_type"] = DOOR_ENV[task_id][0] if task_id < len(DOOR_ENV) else None
            desc["joint_range"] = DOOR_ENV[task_id][1] if task_id < len(DOOR_ENV) else None
        return desc
