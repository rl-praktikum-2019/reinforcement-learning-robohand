# This class should represent the agent-environment interface needed 
# for the setup of every experiment.

import numpy as np
import tensorflow as tf
import tflearn
from gym import make
from ddpg.actor_network import ActorNetwork
from ddpg.critic_network import CriticNetwork
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_buffer import ReplayBuffer
from wrappers.gym_wrapper import ThrowEnvWrapper
import pydmps

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


class ExperimentSetup():
    def __init__(self, method, env_name, sess, random_seed):
        self.method = method
        self.sess = sess
        self.ep_ave_max_q = 0

        self.env = ThrowEnvWrapper(make(env_name, reward_type='dense'))
        self.env.seed(random_seed)

    def setup_experiment(self, args):
        if 'dmp' in self.method:
            self.setup_dmp(args)
        if 'ddpg' in self.method:
            self.setup_ddpg(args)

    # TODO: maybe pass dmp args
    def setup_dmp(self, args=None):
        # 1-dimensional since joint can only move in one axis -> up/down axis
        self.trajectory = [[0], [0], [0], [0], [-1], [1]]
        y_des = np.array(self.trajectory).T
        y_des -= y_des[:, 0][:, None]
        self.dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=600)
        self.dmp.imitate_path(y_des=y_des)

    def setup_ddpg(self, args):
        sess = self.sess
        tf.set_random_seed(int(args['random_seed']))

        # Fetch environment state and action space properties
        state_dim = self.env.observation_space["observation"].shape[0]
        action_dim = self.env.action_space.shape[0]

        #TODO: change action bound to add bias in DDPG learning of actions -> stay close to our default policy and reduce action space
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
        self.replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

        # Needed to enable BatchNorm.
        # This hurts the performance on Pendulum but could be useful
        # in other environments.
        tflearn.is_training(True)

    def update_replay_buffer(self, state, action, next_state, reward, terminal):
        # TODO: Find out what this is and rename it accordingly
        r_state = np.reshape(state, (self.actor.s_dim,))
        r_action = np.reshape(action, (self.actor.a_dim,))
        r_next_state = np.reshape(next_state, (self.actor.s_dim,))
        self.replay_buffer.add(r_state, r_action, reward, terminal, r_next_state)

    def learn_ddpg_minibatch(self, args):
        # Keep adding experience to the memory until there are at least minibatch size samples
        if self.replay_buffer.size() > int(args['minibatch_size']):
            s_batch, a_batch, r_batch, t_batch, s2_batch = \
                self.replay_buffer.sample_batch(int(args['minibatch_size']))

            # Calculate targets
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(int(args['minibatch_size'])):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

            # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(s_batch, a_batch,
                                                     np.reshape(y_i, (int(args['minibatch_size']), 1)))

            self.ep_ave_max_q += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()
