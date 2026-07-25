import gymnasium as gym
import os
import gymnasium.envs.mujoco

# Only the half-cheetah variants are used by the continual-learning tasks
# (see CHEETAH_ENVS in hypercrl/envs/cl_env.py). The gravity / wall / sensor
# variants share the same modified_half_cheetah module and are kept for
# completeness; every other robot (ant, hopper, walker, humanoid, reacher,
# pusher-arm, inverted pendulum) was removed as unused.
custom_envs = {
            "HalfCheetahGravityHalf-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(gravity=-4.905)),
            "HalfCheetahGravityThreeQuarters-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(gravity=-7.3575)),
            "HalfCheetahGravityOneAndHalf-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(gravity=-14.715)),
            "HalfCheetahGravityOneAndQuarter-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahGravityEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(gravity=-12.2625)),

            "HalfCheetahWall-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahWallEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict()),
            "HalfCheetahWithSensor-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahWithSensorEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(model_path=os.path.dirname(gymnasium.envs.mujoco.__file__) + "/assets/half_cheetah.xml")),

            # Modified body parts — HalfCheetah (used by CHEETAH_ENVS tasks 1–4)
            "HalfCheetahBigTorso-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["torso"], size_scale=1.25)),
            "HalfCheetahBigThigh-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["fthigh", "bthigh"], size_scale=1.25)),
            "HalfCheetahBigLeg-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["fshin", "bshin"], size_scale=1.25)),
            "HalfCheetahBigFoot-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["ffoot", "bfoot"], size_scale=1.25)),
            "HalfCheetahSmallTorso-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["torso"], size_scale=.75)),
            "HalfCheetahSmallThigh-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["fthigh", "bthigh"], size_scale=.75)),
            "HalfCheetahSmallLeg-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["fshin", "bshin"], size_scale=.75)),
            "HalfCheetahSmallFoot-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["ffoot", "bfoot"], size_scale=.75)),
            "HalfCheetahSmallHead-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["head"], size_scale=.75)),
            "HalfCheetahBigHead-v0":
                dict(path='hypercrl.envs.mujoco.modified_half_cheetah:HalfCheetahModifiedBodyPartSizeEnv',
                     max_episode_steps=1000,
                     reward_threshold=4800.0,
                     kwargs=dict(body_parts=["head"], size_scale=1.25)),
                     }


def register_custom_envs():
    for key, value in custom_envs.items():
        arg_dict = dict(id=key,
                        entry_point=value["path"],
                        max_episode_steps=value["max_episode_steps"],
                        kwargs=value["kwargs"])

        if "reward_threshold" in value:
            arg_dict["reward_threshold"] = value["reward_threshold"]

        gym.envs.register(**arg_dict)


register_custom_envs()
