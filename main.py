import numpy as np
import gym
import time
from utils.visualization import random_robby_plot, update_plot

episodes = 2
episode_length = 2500
PLOT_FREQUENCY=200

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
rewards=[]
cum_rewards=[]
substitute_rewards=[]

plot=random_robby_plot('random_'+str(episode_length), rewards, cum_rewards)

for j in range(episode_length):
    action = policy(obs['observation'], obs['desired_goal'])
    #print("Ball [X,Y,Z]", get_ball_data(env))
    obs, reward, done, info = env.step(action)
    reward=get_ball_data(env)[1]
    rewards.append(reward)
    if j==0: 
        cum_rewards.append(reward)
    else:
        cum_rewards.append(cum_rewards[j-1]+reward)
    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward(obs['achieved_goal'], substitute_goal, info)
    substitute_rewards.append(substitute_reward)
    #print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))
    #print(info)
    env.render()
    #plot=random_robby_plot('random_'+str(episode_length), rewards, cum_rewards)

    if j % PLOT_FREQUENCY:
        print('Is mod 200!')
        #update_plot(plot)
        plot=random_robby_plot('random_'+str(episode_length), rewards, cum_rewards)


print('Rewards:',rewards) 
print('Cum. Rewards: ',cum_rewards) 
print('Sub. Rewards:',substitute_rewards)
random_robby_plot('random_'+str(episode_length), rewards, cum_rewards)
env.close()
