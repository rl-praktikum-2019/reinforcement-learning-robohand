import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns
import multiprocessing as mp

STEPS = 100
EPISODES = 20
PAUSE = 1e-6


class ProcessPlotter(object):
    def __init__(self, algorithm, total_steps):
        self.x = []
        self.y = []
        self.reward_memory = []
        self.current_episode = 0
        self.total_steps = total_steps
        self.algorithm = algorithm
        self.cum_rewards = []

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.clear_axes()
                self.plot_graphs_per_episode(command)

        self.fig.canvas.draw()
        return True

    def clear_axes(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

    def __call__(self, pipe):
        print('Starting plotting.')

        self.pipe = pipe
        # self.fig, self.ax = plt.subplots()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, num='Performance measures')

        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(0,65,545, 545)

        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.9)

        # self.ax.set_title('Please wait for the end of the first episode.')
        timer = self.fig.canvas.new_timer(interval=500)
        timer.add_callback(self.call_back)
        timer.start()
        plt.show()

    def plot_graphs_per_episode(self, plot_data):
        # TODO: this only works if we call this function per episode NOT per step
        self.current_episode += 1
        rewards, ball_heights = plot_data[0], plot_data[1]
        self.reward_memory.append(rewards)
        self.cum_rewards.append(sum(rewards))

        ci = 0.95  # 95% confidence interval
        #
        # Important:    Setting axis=0 results in the mean of reward at each step not in the episode. Since we save rewards
        #               for each episode (length of array is steps) in the reward memory we have a 2-dim array. We take the
        #               mean of the column not the row!
        #               (row=all rewards in an episode, column=reward at a step over all episodes)
        #
        reward_means = np.mean(self.reward_memory, axis=0)
        stds = np.std(self.reward_memory, axis=0)

        # compute upper/lower confidence bounds
        test_stat = st.t.ppf((ci + 1) / 2, self.current_episode)
        lower_bound = reward_means - test_stat * stds / np.sqrt(self.current_episode)
        upper_bound = reward_means + test_stat * stds / np.sqrt(self.current_episode)

        x = np.arange(0, self.total_steps, 1)

        #
        # First graph - Reward per step of the current episode
        #
        self.ax1.grid()
        self.ax1.plot(rewards, color='orange', label="Last epsilon=%.2f" % 0)
        self.ax1.set_title(
            self.algorithm + ': Current avg. reward per step in experiment %d: %.4f' % (
                self.current_episode, np.mean(rewards)))
        self.ax1.set_ylabel('Reward')
        self.ax1.set_xlabel('Step')
        
        #
        # Second graph - Avg reward per step over all currently measured episodes
        #
        self.ax2.grid()
        self.ax2.plot(reward_means, color='blue', label="Mean epsilon=%.2f" % 0)
        self.ax2.set_title(
            self.algorithm + ': Avg. reward per step in experiment %d: %.4f' % (
                self.current_episode, sum(reward_means) / self.total_steps))
        # plot upper/lower confidence bound
        self.ax2.fill_between(x=x, y1=lower_bound, y2=upper_bound, color='blue', alpha=0.2, label="CI %.2f" % ci)
        self.ax2.set_ylabel('Reward')
        self.ax2.set_xlabel('Step')
        #
        # Third graph - Ball height per step in last episode
        #
        self.ax3.set_ylim(0, 1)
        self.ax3.axhline(y=0.4, alpha=.3, color='g')
        self.ax3.axhline(y=0.5, alpha=.4, color='g')
        self.ax3.plot(ball_heights, color='blue', label='Ball height')

        # plot upper/lower confidence bound
        self.ax3.set_title(
            self.algorithm + ': Avg. Height in episode %d: %.4f' % (self.current_episode, np.mean(ball_heights)))
        self.ax3.set_ylabel('Height')
        self.ax3.set_xlabel('Step')
        plt.legend()

        # plt.subplot_tool()


class Plot(object):
    def __init__(self, algorithm, total_steps):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(algorithm, total_steps)
        self.plot_process = mp.Process(target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

    def plot(self, data, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(data)


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
        plt.figure(figsize=(20, 10))
        plt.xticks(rotation=-45)
        # plot boxplot with seaborn
        bplot = sns.boxplot(y='step_reward', x='configuration',
                            data=step_reward_box,
                            width=0.75,
                            palette="colorblind")
        # add swarmplot
        bplot = sns.swarmplot(y='step_reward', x='configuration',
                              data=step_reward_box,
                              color='black',
                              alpha=0.75)
        return plt
