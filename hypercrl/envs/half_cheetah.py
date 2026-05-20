import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self, render_mode=None):
        self.xposbefore = None
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
        )
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(
            self, 'half_cheetah.xml', 5, observation_space, render_mode=render_mode)

    def step(self, action):
        self.xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        return ob, reward, False, False, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            (self.data.qpos.flat[:1] - self.xposbefore) / self.dt,
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + \
            self.np_random.standard_normal(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.xposbefore = np.copy(self.data.qpos[0])
        return self._get_obs()
