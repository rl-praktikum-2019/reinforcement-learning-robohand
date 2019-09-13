import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pydmps
from wrappers.gym_wrapper import ThrowEnvWrapper
import gym

# TODO: Delete this file when its knowledge is no longer needed.
STEPS = 100
EPISODES = 20
PAUSE = 1e-6


def initialize():
    ## interactive plotting on (no need to close window for next iteration)
    plt.ion()
    plt.grid()
    plt.ylim(20, 20)  # limit y axis
    plt.title("Please wait for the end of the first episode.")


def close():
    ## disable interactive plotting => otherwise window terminates
    plt.ioff()
    plt.show()


def plot_avg_reward_per_step(reward_memory, method_name, episode, steps):
    ci = 0.95  # 95% confidence interval
    # axis=0 results in the mean of reward at each step not in the episode
    reward_means = np.mean(reward_memory, axis=0)
    stds = np.std(reward_memory, axis=0)

    # compute upper/lower confidence bounds
    test_stat = st.t.ppf((ci + 1) / 2, episode)
    lower_bound = reward_means - test_stat * stds / np.sqrt(episode)
    upper_bound = reward_means + test_stat * stds / np.sqrt(episode)

    print('Avg. Reward per step in experiment %d: %.4f' % (episode, sum(reward_means) / steps))

    # clear plot frame
    plt.clf()

    # plot average reward
    ax = plt.plot(reward_means, color='blue', label="epsilon=%.2f" % 0)

    # plot upper/lower confidence bound
    x = np.arange(0, steps, 1)
    ax = plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)
    plt.grid()
    plt.ylim(-20, 20)  # limit y axis
    plt.title(method_name + ': Avg. Reward per step in experiment %d: %.4f' % (episode, sum(reward_means) / steps))
    plt.ylabel("Reward per step")
    plt.xlabel("Play")
    plt.legend()
    plt.draw()
    plt.pause(PAUSE)


def plot_reward_per_step(rewards, method_name, episode, steps):
    print('Avg. Reward in experiment %d: %.4f' % (episode, np.mean(rewards)))

    # clear plot frame
    plt.clf()

    # plot average reward
    ax = plt.plot(rewards, color='blue', label="reward")

    # plot upper/lower confidence bound
    x = np.arange(0, steps, 1)
    plt.grid()
    plt.ylim(-20, 20)  # limit y axis
    plt.title(method_name + ': Avg. Reward in experiment %d: %.4f' % (episode, np.mean(rewards)))
    plt.ylabel("Reward")
    plt.xlabel("Step")
    plt.legend()
    plt.draw()
    plt.pause(PAUSE)


env = ThrowEnvWrapper(gym.make('HandManipulateEgg-v0'))
obs = env.reset()
reward_memory = []

initialize()
for episode in range(EPISODES):

    obs = env.reset()
    episode_rewards = np.zeros(STEPS)
    for step in range(STEPS):
        obs, reward, done, info = env.step(env.action_space.sample())
        episode_rewards[step] = reward
        if done:
            break
        env.render()
        plt.pause(PAUSE)

    reward_memory.append(episode_rewards)
    #plot_avg_reward_per_step(reward_memory, 'DMP', episode + 1, STEPS)

    plot_reward_per_step(episode_rewards, 'DMP', episode + 1, STEPS)

env.close()
close()
