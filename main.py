import numpy as np
import gym
import time

episodes = 1000
episode_length = 20000


def policy(observation, desired_goal):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()


def get_ball_data(env):
    x_pos = env.env.sim.data.get_body_xpos("object")
    x_velp = env.env.sim.data.get_body_xvelp("object")
    x_velr = env.env.sim.data.get_body_xvelr("object")
    return x_pos, x_velp, x_velr


env = gym.make("HandManipulateEgg-v0")

obs = env.reset()

for j in range(episode_length):
    action = policy(obs['observation'], obs['desired_goal'])
    print("Ball [X,Y,Z]", get_ball_data(env))
    obs, reward, done, info = env.step(action)
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))
    env.render()

env.close()
