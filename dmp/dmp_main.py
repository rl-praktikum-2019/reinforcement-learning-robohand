import numpy as np
import gym
from wrappers.gym_wrapper import ThrowEnvWrapper
episodes = 50
episode_length = 200


def get_joint_data(env):
    x_pos = env.env.sim.data.get_body_xpos("robot0:mfknuckle")
    x_velp = env.env.sim.data.get_body_xvelp("robot0:mfknuckle")
    x_velr = env.env.sim.data.get_body_xvelr("robot0:mfknuckle")
    return x_pos, x_velp, x_velr

env = ThrowEnvWrapper(gym.make("ThrowBall-v0"))

for i in range(episodes):

    obs = env.reset()

    for j in range(episode_length):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        xpos, xvelp, xvelr = get_joint_data(env)
        print("POS", xpos)
        env.render()

        if done:
            break

env.close()
