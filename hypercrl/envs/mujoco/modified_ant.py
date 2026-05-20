import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env
import os.path as osp
from .wall_envs import MazeFactory
from .gravity_envs import GravityEnvFactory
from .perturbed_bodypart_env import ModifiedSizeEnvFactory

from gymnasium.envs.mujoco.ant import AntEnv

import os
import gymnasium as gym


AntGravityEnv = lambda *args, **kwargs : GravityEnvFactory(ModifiedAntEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/ant.xml", *args, **kwargs)


AntMaze = lambda *args, **kwargs : MazeFactory(ModifiedAntEnv)(model_path=os.path.dirname(gym.envs.mujoco.__file__) + "/assets/ant.xml", ori_ind=0, *args, **kwargs)




class ModifiedAntEnv(AntEnv, utils.EzPickle):
    """
    Simply allows changing of XML file, probably not necessary if we pull request the xml name as a kwarg in openai gym
    """
    def __init__(self, **kwargs):
        mujoco_env.MujocoEnv.__init__(self, kwargs["model_path"], 4)
        utils.EzPickle.__init__(self)
