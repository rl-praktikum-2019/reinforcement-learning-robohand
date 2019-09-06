import numpy as np
import gym
import time
from plots.visualization import random_robby_plot
from wrappers.gym_wrapper import ThrowEnvWrapper
episodes = 50
episode_length = 200


def get_ball_data(env):
    x_pos = env.env.sim.data.get_body_xpos("object")
    x_velp = env.env.sim.data.get_body_xvelp("object")
    x_velr = env.env.sim.data.get_body_xvelr("object")
    return x_pos, x_velp, x_velr


env = ThrowEnvWrapper(gym.make("HandManipulateEgg-v0"))

for i in range(episodes):

    obs = env.reset()
    rewards=[]
    cum_rewards=[]
    substitute_rewards=[]

    for j in range(episode_length):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if j==0:
            cum_rewards.append(reward)
        else:
            cum_rewards.append(cum_rewards[j-1]+reward)
        env.render()

        if done:
            break

    print('Rewards:',rewards)
    print('Cum. Rewards: ',cum_rewards)
    print('Sub. Rewards:',substitute_rewards)
    random_robby_plot('random_'+str(episode_length), rewards, cum_rewards)

env.close()
