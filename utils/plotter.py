import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import gym
from wrappers.gym_wrapper import ThrowEnvWrapper
import seaborn as sns

STEPS = 100
EPISODES = 20
PAUSE = 1e-6


class Plotter():
    def __init__(self):
        self.cum_rewards = []
        self.rewards = []
        self.ball_heights = []
        self.cum_reward = 0

    def initialize(self):
        ## interactive plotting on (no need to close window for next iteration)
        plt.ion()
        plt.grid()
        plt.ylim(20, 20)  # limit y axis
        plt.title("Please wait for the end of the first episode.")

    def clear_episode_data(self):
        self.cum_rewards = []
        self.rewards = []
        self.ball_heights = []
        self.cum_reward = 0

    def close(self):
        ## disable interactive plotting => otherwise window terminates
        plt.ioff()
        plt.show()

    def boxplot(step_reward_box):
        plt.figure(figsize=(20,10))
        plt.xticks(rotation=-45)
        # plot boxplot with seaborn
        bplot=sns.boxplot(y='step_reward', x='configuration',
                        data=step_reward_box,
                        width=0.75,
                        palette="colorblind")
        # add swarmplot
        bplot=sns.swarmplot(y='step_reward', x='configuration',
                    data=step_reward_box,
                    color='black',
                    alpha=0.75)
        return plt

    def plot_cum_reward_per_step(self, reward, algorithm, episode):
        self.cum_reward += reward
        self.cum_rewards.append(self.cum_reward)

        # clear plot frame
        plt.clf()

        # plot average reward
        ax = plt.plot(self.cum_rewards, color='blue', label="reward")
        plt.grid()
        plt.title(algorithm + ': Cum. Reward in experiment %d: %.4f' % (episode, self.cum_reward))
        plt.ylabel("Cumulated Reward")
        plt.xlabel("Step")
        plt.legend()
        plt.draw()
        plt.pause(PAUSE)

    def plot_avg_reward_per_step(self, reward_memory, algorithm, episode, steps):
        ci = 0.95  # 95% confidence interval
        #
        # Important:    Setting axis=0 results in the mean of reward at each step not in the episode. Since we save rewards
        #               for each episode (length of array is steps) in the reward memory we have a 2-dim array. We take the
        #               mean of the column not the row!
        #               (row=all rewards in an episode, column=reward at a step over all episodes)
        #
        reward_means = np.mean(reward_memory, axis=0)
        stds = np.std(reward_memory, axis=0)

        # compute upper/lower confidence bounds
        test_stat = st.t.ppf((ci + 1) / 2, episode)
        lower_bound = reward_means - test_stat * stds / np.sqrt(episode)
        upper_bound = reward_means + test_stat * stds / np.sqrt(episode)

        # clear plot frame
        plt.clf()
        plt.cla()
        plt.close()

        # plot average reward
        #fig = plt.figure(constrained_layout=True)

        fig, (ax1, ax2) = plt.subplots(2)
        #ax = fig.add_subplot(111)    # The big subplot
        #ax1 = fig.add_subplot(211)
        #ax2 = fig.add_subplot(212)

        # plot upper/lower confidence bound
        x = np.arange(0, steps, 1)

        #ax.spines['top'].set_color('none')
        #ax.spines['bottom'].set_color('none')
        #ax.spines['left'].set_color('none')
        #ax.spines['right'].set_color('none')
        #ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

        #ax.set_ylabel("Reward per step")
        #ax.set_xlabel("Steps")

        ax1.grid()
        ax1.plot(reward_memory[-1], color='orange', label="Last epsilon=%.2f" % 0)
        ax1.set_title(algorithm + ': Last reward per step in experiment %d: %.4f' % (episode, sum(reward_means) / steps))

        ax2.grid()
        ax2.plot(reward_means, color='blue', label="Mean epsilon=%.2f" % 0)
        ax2.set_title(algorithm + ': Avg. reward per step in experiment %d: %.4f' % (episode, sum(reward_means) / steps))
        ax2 = plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)

        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
        #plt.subplot_tool()
        plt.ylabel("                                                 Reward")
        plt.xlabel("Steps")
        plt.draw()
        plt.pause(PAUSE)

    def plot_reward_per_step(self, rewards, algorithm, episode, steps):
        print('Avg. Reward in experiment %d: %.4f' % (episode, np.mean(rewards)))

        # clear plot frame
        plt.clf()
        # plot average reward
        ax = plt.plot(rewards, color='blue', label="reward")

        # plot upper/lower confidence bound
        plt.grid()
        plt.title(algorithm + ': Avg. Reward in experiment %d: %.4f' % (episode, np.mean(rewards)))
        plt.ylabel("Reward")
        plt.xlabel("Step")
        plt.legend()
        plt.draw()
        plt.pause(PAUSE)

    def plot_ballheight_per_step(self, env, method_name, episode):
        self.ball_heights.append(env.ball_center_z)
        # clear plot frame
        plt.clf()

        # plot average reward
        ax = plt.plot(self.ball_heights, color='blue', label="reward")

        # plot upper/lower confidence bound
        plt.grid()
        plt.title(method_name + ': Avg. Height in experiment %d: %.4f' % (episode, np.mean(self.ball_heights)))
        plt.ylabel("Height")
        plt.xlabel("Step")
        plt.legend()
        plt.draw()
        plt.pause(PAUSE)

# env = ThrowEnvWrapper(gym.make('ThrowBall-v0'))
# obs = env.reset()
# reward_memory = []
# plotter = Plotter()
#
# plotter.initialize()
# for episode in range(EPISODES):
#
#     obs = env.reset()
#     episode_rewards = np.zeros(STEPS)
#     for step in range(STEPS):
#         obs, reward, done, info = env.step(env.action_space.sample())
#         #plotter.plot_cum_reward_per_step(reward, 'DMP', episode + 1)
#         episode_rewards[step] = reward
#         if done:
#             break
#         env.render()
#
#     #plotter.clear_episode_data()
#     reward_memory.append(episode_rewards)
#     plotter.plot_avg_reward_per_step(reward_memory, 'DMP', episode + 1, STEPS)
#
#     #plotter.plot_reward_per_step(episode_rewards, 'DMP', episode + 1, STEPS)
#
# env.close()
# plotter.close()
