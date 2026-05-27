import numpy as np
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import mujoco_env


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