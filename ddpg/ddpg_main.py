import os
import tensorflow as tf
import tflearn
import numpy as np
import gym
from gym import wrappers
import argparse
import pprint as pp
import math
from ddpg.actor_network import ActorNetwork
from ddpg.critic_network import CriticNetwork
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_buffer import ReplayBuffer
from wrappers.observation_wrapper import ObservationWrapper
from wrappers.reward_wrappers import VelocityRewardWrapper
from wrappers.gym_wrapper import ThrowEnvWrapper
from utils.visualization import random_robby_plots, update_plot

""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf
    
The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow
Author: Patrick Emami
"""
RESULTS_PATH=os.path.dirname(os.path.realpath(__file__))+'/../data/'
PLOT_FREQUENCY=200


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


def train(sess, env, args, actor, critic, actor_noise):
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

    rewards=[]
    cum_rewards=[]
    episode_length=int(args['max_episode_len'])
    cum_plot = random_robby_plots('random_'+str(episode_length), rewards, cum_rewards)

    for i in range(int(args['max_episodes'])):
        print("------------------------ Start episode number:", i)
        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(episode_length):
            print("Start episode step:", j + 1)

            if args['render_env']:
                env.render()

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
            s2, reward, terminal, info = env.step(a[0])

            rewards.append(reward)
            if j==0:
                cum_rewards.append(reward)
            else:
                cum_rewards.append(cum_rewards[j-1]+reward)

            if j % PLOT_FREQUENCY:
                update_plot(cum_plot,'random_'+str(episode_length), cum_rewards)
            
            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), reward,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2

            if not math.isnan(reward):
                ep_reward += reward

            if terminal:
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(j + 1)
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), i,
                                                                             (ep_ave_max_q / float(j + 1))))
                break


def main(args):
    with tf.Session() as sess:

        env = ObservationWrapper(gym.make(args['env'], reward_type='dense'))
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

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

        if args['use_gym_monitor']:
            if not args['render_env']:
                env = wrappers.Monitor(
                    env, args['monitor_dir'], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        if args['use_gym_monitor']:
            env.monitor.close()

        env.close()


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
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default=RESULTS_PATH+'./ddpg_results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default=RESULTS_PATH+'./ddpg_results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
