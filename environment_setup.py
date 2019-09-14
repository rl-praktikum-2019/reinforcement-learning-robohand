# This class should represent the agent-environment interface needed 
# for the setup of every experiment.

import numpy as np
import tensorflow as tf
import tflearn
from ddpg.actor_network import ActorNetwork
from ddpg.critic_network import CriticNetwork
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_buffer import ReplayBuffer


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

class EnvironmentSetup():
    def __init__(self, method, env):
        self.env=env
        self.method=method
    
    def setup_ddpg(self, args):
        with tf.Session() as sess:
            tf.set_random_seed(int(args['random_seed']))

            # Fetch environment state and action space properties
            state_dim = self.env.observation_space["observation"].shape[0]
            action_dim = self.env.action_space.shape[0]
            action_bound = self.env.action_space.high

            # Ensure action bound is symmetric
            assert (all(self.env.action_space.high - self.env.action_space.low))

            self.actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                                float(args['actor_lr']), float(args['tau']),
                                int(args['minibatch_size']))

            self.critic = CriticNetwork(sess, state_dim, action_dim,
                                float(args['critic_lr']), float(args['tau']),
                                float(args['gamma']),
                                self.actor.get_num_trainable_vars())

            self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

            # Set up summary Ops
            summary_ops, summary_vars = build_summaries()

            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

            # Initialize target network weights
            self.actor.update_target_network()
            self.critic.update_target_network()

            # Initialize replay memory
            replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

            # Needed to enable BatchNorm.
            # This hurts the performance on Pendulum but could be useful
            # in other environments.
            tflearn.is_training(True)

            self.sess=sess

            #return sess, actor, critic, actor_noise