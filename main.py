import tensorflow as tf
import pprint as pp
import numpy as np
import argparse
import os
# TODO: Remove redundancy
from gym import wrappers
import gym
from ddpg.ddpg_main import main as run_ddpg_robby
from dmp.dmp_main import main as run_dmp_robby
from dmp.dmp_ddpg_main import main as run_dmp_ddpg_robby

from wrappers.observation_wrapper import ObservationWrapper

RESULTS_PATH=os.path.dirname(os.path.realpath(__file__))+'/../data/'
PLOT_FREQUENCY=200

# TODO: take --env=HandManipulateEgg-v0 out of args 
def build_environment(random_seed, reward):
    env = ObservationWrapper(gym.make(args['env'], reward_type=reward))
    env.seed(random_seed)
    return env

def run_dmp_experiment(experiment_params):
    print('INFO: Running dmp Robby.')
    run_dmp_robby(args)


def run_dmp_ddpg_experiment(experiment_params):
    print('INFO: Running dmp ddpg Robby.')
    run_dmp_ddpg_robby(args)


def run_ddpg_experiment(args):
    print('INFO: Running ddpg Robby.')
    run_ddpg_robby(args)

def run_experiment():
    env=build_environment(random_seed, 'dense')
    run_ddpg_experiment(env)

def main(args):
    random_seed=int(args['random_seed'])

    with tf.Session() as sess:
        env = build_environment(random_seed, 'dense')

        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)

       # Fetch environment state and action space properties
        state_dim = env.observation_space["observation"].shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        # Ensure action bound is symmetric
        assert (all(env.action_space.high - env.action_space.low))

# XXX: Parameters maybe to main?
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

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
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default=RESULTS_PATH+'./ddpg_results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default=RESULTS_PATH+'./ddpg_results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    #main(args)
    #run_dmp_ddpg_experiment(args)
    #run_dmp_experiment(args)
    run_ddpg_experiment(args)