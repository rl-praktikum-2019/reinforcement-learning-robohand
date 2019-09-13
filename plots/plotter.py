import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pydmps
from wrappers.gym_wrapper import ThrowEnvWrapper
import gym

# TODO: Delete this file when its knowledge is no longer needed.
STEPS = 200
EPISODES = 20
PAUSE = 1e-6


def initialize():
    ## interactive plotting on (no need to close window for next iteration)
    plt.ion()
    plt.grid()
    plt.ylim(0, 2)  # limit y axis
    plt.title("Please wait for the end of the first episode.")
    plt.ylabel("Reward per step")
    plt.xlabel("Play")


def close():
    ## disable interactive plotting => otherwise window terminates
    plt.ioff()
    plt.show()


def plot(reward_memory, method_name, episode, steps):
    ci = 0.95  # 95% confidence interval
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
    plt.ylim(0, 2)  # limit y axis
    plt.title(method_name + ': Avg. Reward per step in experiment %d: %.4f' % (episode, sum(reward_means) / steps))
    plt.ylabel("Reward per step")
    plt.xlabel("Play")
    plt.legend()
    plt.draw()
    plt.pause(PAUSE)


env = ThrowEnvWrapper(gym.make('HandManipulateEgg-v0'))
obs = env.reset()
reward_memory = []

initialize()
for episode in range(EPISODES):

    obs = env.reset()
    rewards = []
    for step in range(STEPS):
        obs, reward, done, info = env.step(env.action_space.sample())
        rewards.append(reward)
        if done:
            break
        env.render()
        plt.pause(PAUSE)

    reward_memory.append(rewards)

    plot(reward_memory, 'DMP', episode + 1, STEPS)


env.close()
close()
