import numpy as np
import os.path as osp
import os
import gymnasium as gym

from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
from .wall_envs import WallEnvFactory
from .gravity_envs import GravityEnvFactory
from .perturbed_bodypart_env import ModifiedSizeEnvFactory

from hypercrl.envs.half_cheetah import HalfCheetahEnv

HalfCheetahWallEnv = lambda *args, **kwargs : WallEnvFactory(ModifiedHalfCheetahEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/half_cheetah.xml", ori_ind=-1, *args, **kwargs)

HalfCheetahGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedHalfCheetahEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/half_cheetah.xml", *args, **kwargs)

HalfCheetahModifiedBodyPartSizeEnv = lambda *args, **kwargs : ModifiedSizeEnvFactory(ModifiedHalfCheetahEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/half_cheetah.xml", *args, **kwargs)

class ModifiedHalfCheetahEnv(HalfCheetahEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, render_mode=None, **kwargs):
        self.xposbefore = None
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5, observation_space, render_mode=render_mode)

class HalfCheetahWithSensorEnv(HalfCheetahEnv, utils.EzPickle):
    """
    Adds empty sensor readouts, this is to be used when transfering to WallEnvs where we get sensor readouts with distances to the wall
    """

    def __init__(self, n_bins=10, render_mode=None, **kwargs):
        self.n_bins = n_bins
        observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18 + n_bins,), dtype=np.float64)
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 5, observation_space, render_mode=render_mode)


    def _get_obs(self):
        obs = np.concatenate([
            HalfCheetahEnv._get_obs(self),
            np.zeros(self.n_bins)
            # goal_readings
        ])
        return obs
