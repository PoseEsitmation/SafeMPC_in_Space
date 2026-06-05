import gymnasium
import matplotlib.pyplot as plt
from IPython import display
from mpl_toolkits.mplot3d import Axes3D
from hypercrl.envs.sat_env import SatDynEnv

# Create the environment
env = SatDynEnv()

# Start the environment
obs, _ = env.reset()

"""
import matplotlib.pyplot as plt
from IPython import display

# Start the environment
state = env.reset()

img = plt.imshow(env.render(mode='rgb_array'))  # Only call this once

for _ in range(1000):
    img.set_data(env.render(mode='rgb_array'))  # Just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)
    
    # Take a random action
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    
    if done:
        state = env.reset()
"""