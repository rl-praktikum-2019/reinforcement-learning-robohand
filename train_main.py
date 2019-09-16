import pprint as pp
import math
import numpy as np
import tensorflow as tf
import argparse
import os
from experiment_setup import ExperimentSetup
from gym import make
from utils.plots import init_cum_reward_plot, update_plot

RESULTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'


def compute_action(setup, state=None, method=None):
    action = None
    if 'ddpg' in method:
        action = (setup.actor.predict(np.reshape(state, (1, setup.actor.s_dim))) +
                  setup.actor_noise())[0]

    if 'dmp' in method:
        # TODO: better dmp approach
        # 1. remove wrist joint from learning
        # 2. overwrite but learn wrist (current/first solution)
        # 3. use dmp as bias, try to learn dmp value
        y_track, dy_track, ddy_track = setup.dmp.step()

        if action is None:
            # Reduce the effect of dmp trajectory for other joints (fingers)
            clipped_ddy = np.clip(ddy_track, -0.5, 0.5)
            action = np.full((20,), clipped_ddy[0])
            # Remove action for vertical wrist joint
            action[0] = 0

        action[1] = ddy_track[0]

    # TODO: notify error method needed
    if action is None:
        action = np.zeros(20)

    assert action is not None
    return action


def train_experiment(method, setup):
    env = setup.env
    print('INFO: Training for ' + method)
    episode_length = int(args['max_episode_len'])

    if 'dmp' in method:
        episode_length = setup.dmp.timesteps

    rewards = []
    cum_rewards = []

    for episode in range(int(args['max_episodes'])):
        rewards.clear()
        cum_rewards.clear()
        ep_reward = 0

        state = env.reset()
        if 'dmp' in method:
            setup.dmp.reset_state()

        # TODO: see plot class todos
        if args['plot']:
            cum_plot = init_cum_reward_plot('random_' + str(episode_length), rewards, cum_rewards)

        for step in range(episode_length):
            if args['render_env']:
                env.render()
            action = compute_action(setup, state, method)

            # TODO: adapt reward to throw task at a different location
            next_state, reward, terminal, info = env.step(action)

            rewards.append(reward)
            cum_rewards.append(np.sum(rewards))

            if (step % int(args['plot_frequency'])) and args['plot']:
                update_plot(cum_plot, 'random_' + str(episode_length), cum_rewards)

            if 'ddpg' in method:
                setup.update_replay_buffer(state, action, next_state, reward, terminal)

                # XXX: Select only args we need instead of all args
                setup.learn_ddpg_minibatch(args)

                # NOTE: Important for DDPG actor prediction!
                state = next_state

            if not math.isnan(reward):
                ep_reward += reward

            if terminal:
                # TODO: Save performance metrics in separate class and print them from there
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), episode,
                                                                             (setup.ep_ave_max_q / float(step + 1))))
                break


def main(args):
    random_seed = int(args['random_seed'])
    method = args['method']

    with tf.Session() as sess:
        print(args['env'])
        exp_setup = ExperimentSetup(method, args['env'], sess, args['random_seed'])
        exp_setup.setup_experiment(args)
        train_experiment(method, exp_setup)


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
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='HandManipulateEgg-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results',
                        default=RESULTS_PATH + './ddpg_results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default=RESULTS_PATH + './ddpg_results/tf_ddpg')
    parser.add_argument('--method',
                        help="reinforcement learning method for experiment. Possible values are: 'ddpg', 'dmp', 'dmp_ddpg'",
                        default='dmp_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
    # run_dmp_ddpg_experiment(args)
    # run_dmp_experiment(args)
    # run_ddpg_experiment(args)
