import tensorflow as tf
import pprint as pp
import numpy as np
import argparse
import os
from utils.replay_buffer import  ReplayBuffer
import tflearn
from ddpg.actor_network import ActorNetwork
from ddpg.critic_network import CriticNetwork
from utils.noise import OrnsteinUhlenbeckActionNoise

# TODO: Remove redundancy
from gym import wrappers
import gym
# from ddpg.ddpg_main import main as run_ddpg_robby
# from dmp.dmp_main import main as run_dmp_robby
# from dmp.dmp_ddpg_main import main as run_dmp_ddpg_robby

from wrappers.observation_wrapper import ObservationWrapper
from utils.plots import init_cum_reward_plot, update_plot
import math

RESULTS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'


# TODO: take --env=HandManipulateEgg-v0 out of args
def build_environment(random_seed, reward):
    env = ObservationWrapper(gym.make(args['env'], reward_type=reward))
    env.seed(random_seed)
    return env


# def run_dmp_experiment(experiment_params):
#     print('INFO: Running dmp Robby.')
#     run_dmp_robby(args)
#
#
# def run_dmp_ddpg_experiment(experiment_params):
#     print('INFO: Running dmp ddpg Robby.')
#     run_dmp_ddpg_robby(args)
#
#

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# TODO: setup class containing all needed resources for running experiment
def setup_ddpg(args, env):
    with tf.Session() as sess:

        tf.set_random_seed(int(args['random_seed']))

        # Fetch environment state and action space properties
        state_dim = env.observation_space["observation"].shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        # Ensure action bound is symmetric
        assert (all(env.action_space.high - env.action_space.low))

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # Set up summary Ops
        summary_ops, summary_vars = build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()

        # Initialize replay memory
        replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        tflearn.is_training(True)

        return sess, actor, critic, actor_noise

# def run_ddpg_experiment():
#     print('INFO: Running ddpg Robby.')
#     run_ddpg_robby(args)


def compute_action(actor=None, state=None, actor_noise=None, method=None):
    action = None
    if method == 'ddpg':
        action = (actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise())[0]
    else:
        action = np.zeros(20)
        #TODO: notify error method needed

    return action


def train_experiment(env):

    episode_length = int(args['max_episode_len'])

    for episode in range(int(args['max_episodes'])):

        rewards = []
        cum_rewards = []
        ep_reward = 0
        ep_ave_max_q = 0

        state = env.reset()

        #TODO: see plot class todos
        if args['plot']:
            cum_plot = init_cum_reward_plot('random_'+str(episode_length), rewards, cum_rewards)

        for step in range(episode_length):
            action = compute_action()  # TODO: get action according to method

            # TODO: reward changes at a different location
            next_state, reward, terminal, info = env.step(action)

            rewards.append(reward)
            cum_rewards.append(np.sum(rewards))

            if (step % int(args['plot_frequency'])) and args['plot']:
                update_plot(cum_plot,'random_'+str(episode_length), cum_rewards)

            # TODO: add replay buffer for DDPG
            # TODO: add minibatch learning for DDPG

            # NOTE: Important for DDPG actor prediction!
            state = next_state

            if not math.isnan(reward):
                ep_reward += reward

            if terminal:
                # TODO: print Qmax for ddpg
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), episode,
                                                                             (ep_ave_max_q / float(step + 1))))
                break


def main(args):
    random_seed = int(args['random_seed'])
    env = build_environment(random_seed, 'dense')
    train_experiment(env)


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
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results',
                        default=RESULTS_PATH + './ddpg_results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default=RESULTS_PATH + './ddpg_results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
    # run_dmp_ddpg_experiment(args)
    # run_dmp_experiment(args)
    #run_ddpg_experiment(args)
