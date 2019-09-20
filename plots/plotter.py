import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import gym
from wrappers.gym_wrapper import ThrowEnvWrapper
import multiprocessing as mp
import time
import pickle

STEPS = 100
EPISODES = 20
PAUSE = 1e-6


class ProcessPlotter(object):
    def __init__(self):
        self.x = []
        self.y = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.ax.clear()

                self.plot_reward_per_step(command)

        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('Starting plotting.')

        self.pipe = pipe
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Please wait for the end of the first episode.")
        timer = self.fig.canvas.new_timer(interval=500)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()

    def plot_reward_per_step(self, command):
        data_array = command[:]
        self.x = np.arange(0, len(data_array), 1)
        self.y = data_array
        plt.title('Reward per step')
        plt.ylabel("Reward")
        plt.xlabel("Step")
        self.ax.plot(self.x, self.y, color='blue', label="reward")
        plt.legend()


class NBPlot(object):
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, data, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(data)


def main():
    env = ThrowEnvWrapper(gym.make('ThrowBall-v0'))
    obs = env.reset()
    reward_memory = []
    episode_rewards = np.zeros(STEPS)
    pl = NBPlot()
    for episode in range(EPISODES):

        obs = env.reset()
        episode_rewards[:] = 0
        for step in range(STEPS):
            obs, reward, done, info = env.step(env.action_space.sample())
            episode_rewards[step] = reward
            if done:
                break
            env.render()
        pl.plot(episode_rewards)

        reward_memory.append(episode_rewards)

    pl.plot(finished=True)
    env.close()


if __name__ == '__main__':
    if plt.get_backend() == "MacOSX":
        mp.set_start_method("forkserver")
    main()


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

        # plot average reward
        ax = plt.plot(reward_means, color='blue', label="epsilon=%.2f" % 0)

        # plot upper/lower confidence bound
        x = np.arange(0, steps, 1)
        ax = plt.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)
        plt.grid()
        plt.title(algorithm + ': Avg. Reward per step in experiment %d: %.4f' % (episode, sum(reward_means) / steps))
        plt.ylabel("Reward per step")
        plt.xlabel("Play")
        plt.legend()
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
