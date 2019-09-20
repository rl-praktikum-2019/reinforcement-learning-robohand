import pprint as pp
import math
import numpy as np
import tensorflow as tf
import argparse
import os
from experiment_setup import ExperimentSetup
from gym import make
from utils.plots import init_cum_reward_plot, update_plot
from utils.plotter import Plotter

RESULTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'


def compute_action(setup, state=None, algorithm=None):
    action = None
    if 'ddpg' in algorithm and 'dmp' not in algorithm:
        return (setup.actor.predict(np.reshape(state, (1, setup.actor.s_dim))) +
                setup.actor_noise())[0]

    if 'dmp' in algorithm and 'ddpg' not in algorithm:
        y_track, dy_track, ddy_track = setup.dmp.step()

        action = np.full((20,), ddy_track[0])
        # Remove action for horizontal wrist joint
        action[0] = 0
        return action, None

    if 'dmp' in algorithm and 'ddpg' in algorithm:
        y_track, dy_track, ddy_track = setup.dmp.step()

        dmp_action = np.full((20,), ddy_track[0])
        # Remove action for horizontal wrist joint
        dmp_action[0] = 0

        ddpg_action = (setup.actor.predict(np.reshape(state, (1, setup.actor.s_dim))) +
                       setup.actor_noise())[0]

        return dmp_action, ddpg_action

    assert action is not None
    return action, None


def train_experiment(algorithm, setup):
    env = setup.env
    writer = tf.summary.FileWriter(args['summary_dir'], setup.sess.graph)

    print('INFO: Training for ' + algorithm)
    episode_length = int(args['max_episode_len'])

    plotter = Plotter()
    plotter.initialize()

    if 'dmp' in algorithm:
        episode_length = 20

    reward_memory = []
    cum_rewards = []

    for episode in range(int(args['max_episodes'])):
        # rewards.clear()
        # cum_rewards.clear()
        ep_reward = 0
        rewards = np.zeros(episode_length)

        state = env.reset()

        if 'dmp' in algorithm:
            setup.dmp.reset_state()

        # TODO: see plot class todos
        #if args['plot']:
            #cum_plot = init_cum_reward_plot('random_' + str(episode_length), rewards, cum_rewards)

        for step in range(episode_length):
            if args['render_env']:
                env.render()

            action, ddpg_action = compute_action(setup, state, algorithm)

            # TODO: adapt reward to throw task at a different location
            if ddpg_action is not None:
                action += ddpg_action
            next_state, reward, terminal, info = env.step(action)

            rewards[step] = reward
            cum_rewards.append(np.sum(rewards))

            # if (step % int(args['plot_frequency'])) and args['plot']:
            # update_plot(cum_plot, 'random_' + str(episode_length), cum_rewards)

            if 'ddpg' in algorithm:
                setup.update_replay_buffer(state, ddpg_action, next_state, reward, terminal)

                # XXX: Select only args we need instead of all args
                setup.learn_ddpg_minibatch(args)

                # NOTE: Important for DDPG actor prediction!
                state = next_state

            if not math.isnan(reward):
                ep_reward += reward

            if terminal:
                summary_str = setup.sess.run(setup.summary_ops, feed_dict={
                    setup.summary_vars[0]: ep_reward,
                    setup.summary_vars[1]: setup.ep_ave_max_q / float(step)
                })

                writer.add_summary(summary_str, episode)
                writer.flush()

                # TODO: Save performance metrics in separate class and print them from there
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), episode,
                                                                             (setup.ep_ave_max_q / float(step + 1))))
                break
        reward_memory.append(rewards)
        plotter.plot_avg_reward_per_step(reward_memory, algorithm, episode + 1, episode_length)
        # plotter.plot_reward_per_step(rewards, method, episode, episode_length)
        # plotter.clear_episode_data()

    plotter.close()
    env.close()


def main(args):
    algorithm = args['algo']

    with tf.Session() as sess:
        print(args['env'])
        exp_setup = ExperimentSetup(algorithm, args['env'], sess, args['random_seed'])
        exp_setup.setup_experiment(args)

        train_experiment(algorithm, exp_setup)


# XXX: Parameters maybe to main?
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # plot parameters
    parser.add_argument('--plot', help='plot performance measures of agent', action='store_true')
    parser.add_argument('--plot-frequency', help='plot frequency', default=200)

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.01)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='HandManipulateEgg-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results',
                        default=RESULTS_PATH + './ddpg_results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default=RESULTS_PATH + './ddpg_results/tf_ddpg')
    parser.add_argument('--algo',
                        help="reinforcement learning algo for experiment. Possible values are: 'ddpg', 'dmp', 'dmp_ddpg'",
                        default='dmp_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
    # run_dmp_ddpg_experiment(args)
    # run_dmp_experiment(args)
    # run_ddpg_experiment(args)
