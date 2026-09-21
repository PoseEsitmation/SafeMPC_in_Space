import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
from gymnasium.wrappers import TimeLimit
from hypercrl.envs.mujoco.half_cheetah_safe import HalfCheetahSafeEnv
from hypercrl.envs.space_tasks import FAMILIES as SPACE_FAMILIES, get_task_spec

# Per-task gravity rotations for the half_cheetah (body/gravity) continual tasks.
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
CARTPOLE_ENVS = ['MBRLCartpole-v0', 'CartpoleLong1-v0', 'CartpoleShort1-v0',
                 'CartpoleLong2-v0', 'CartpoleLong3-v0', 'CartpoleLong4-v0',
                 'CartpoleShort2-v0', 'CartpoleLong5-v0', 'CartpoleLong6-v0',
                 'CartpoleLong7-v0']
CARTPOLE_BIN_ENVS = ['MBRLCartpole-v0',
                     'CartpoleLeft1-v0', 'CartpoleRight1-v0']


class EnvSpecs():
    a_dims = {
        "half_cheetah": 6,
        "half_cheetah_body": 6,
        "half_cheetah_safe": 6,
        "cartpole": 1,
        "cartpole_bin": 1,
        **{name: 3 for name in SPACE_FAMILIES},
    }

    x_dims = {
        "half_cheetah": 18,
        "half_cheetah_body": 18,
        "half_cheetah_safe": 19,
        "cartpole": 4,
        "cartpole_bin": 4,
        **{name: 13 for name in SPACE_FAMILIES},
    }

    @classmethod
    def get_dim_unit(cls, env):
        return [" " for _ in range(cls.x_dims[env])]

    @classmethod
    def get_dim_name(cls, env):
        return [f"Dim {i+1}" for i in range(cls.x_dims[env])]


class CLEnvHandler():
    def __init__(self, env, seed, seed_offset=0):
        self.cl_env = env
        self.seed = seed
        self.seed_offset = seed_offset  # separates e.g. training and eval envs

        self._envs = []

    def add_task(self, task_id, render=False, replica=False):
        if self.cl_env == "half_cheetah_body":
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
        elif self.cl_env == "cartpole_bin":
            env = gym.make(
                CARTPOLE_BIN_ENVS[task_id], render_mode="human" if render else None)
        elif self.cl_env == "cartpole":
            env = gym.make(CARTPOLE_ENVS[task_id],
                           render_mode="human" if render else None)
        elif self.cl_env in SPACE_FAMILIES:
            from .space_KOZ import SatDynEnv
            env = SatDynEnv(**get_task_spec(self.cl_env, task_id).env_kwargs(),
                            render_mode="human" if render else None)
        else:
            raise ValueError(f"Unknown environment: {self.cl_env}")

        if hasattr(env, 'seed') and self.seed is not None:
            env.seed(self.seed + 7919 * task_id + self.seed_offset)

        if not replica:
            self._envs.append(env)
            return self.get_env(task_id)
        else:
            return env

    def get_env(self, task_id):
        assert 0 <= task_id < len(self._envs)
        return self._envs[task_id]

    def close(self):
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
        elif env_name in ("half_cheetah_body", "half_cheetah"):
            desc["gym_id"] = CHEETAH_ENVS[task_id] if task_id < len(CHEETAH_ENVS) else None
        elif env_name == "half_cheetah_safe":
            desc["keep_out_zones"] = str(HALF_CHEETAH_SAFE_ENVS[task_id]) if task_id < len(HALF_CHEETAH_SAFE_ENVS) else None
        elif env_name in SPACE_FAMILIES:
            desc["params"] = get_task_spec(env_name, task_id).summary()
        return desc
